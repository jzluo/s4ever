// Return exp(x) - 1, given ex = exp(x). For x <= 1, the function computes 2 t / (1 - t) with
// t = tanh(x / 2), which does not cancel near 0. For x > 1, it returns ex - 1, which is accurate
// there, while 1 - t cancels as t approaches 1.
inline float exp_m1(float x, float ex) {
    float t = precise::tanh(x / 2);
    return x > 1 ? ex - 1 : 2 * t / (1 - t);
}
// Return log(1 + x). The factor x / (y - 1) corrects for the rounding error of y = 1 + x.
inline float log_1p(float x) {
    float y = 1 + x;
    return y == 1 ? x : precise::log(y) * x / (y - 1);
}

// Return z^e = exp(e w) for w = log z. The function reduces the angle e Im(w) by multiples of
// 2 pi = PI2 + PI2_LO and adds back lo, the rounding error of the product e Im(w), so that the
// reduced angle stays accurate for large e.
inline float2 power(uint e, float2 w) {
    float x = float(e) * w.y, lo = fma(float(e), w.y, -x);
    float k = floor(x * (1 / (2 * M_PI_F)) + 0.5f);
    float r = fma(-k, PI2_LO, fma(-k, PI2, x)) + lo;
    float c, s = precise::sincos(r, c);
    return precise::exp(float(e) * w.x) * float2(c, s);
}

// Set w = log dA and dB for one mode.
inline void discretize(float dt, float2 A, float2 B, thread float2& w, thread float2& dB) {
    float2 a = dt * A, q;
    if (ZOH) {
        // log dA = a = dt A and dB = B expm1(a) / A. The real part of expm1(a) is computed as
        // expm1(Re a) cos(Im a) - 2 sin(Im a / 2)^2, which does not cancel for small a.
        float s = precise::sin(a.y / 2), ex = precise::exp(a.x);
        float2 e = float2(exp_m1(a.x, ex) * precise::cos(a.y) - 2 * s * s,
                          ex * precise::sin(a.y));
        q = cmulc(e, A) / dot(A, A);
        w = a;
    } else {
        // For p = a/2, log dA = 2 atanh(p) and dB = B dt / (1 - p). The atanh follows
        // `ops._complex.atanh`.
        float2 p = a / 2;
        float den = (1 - p.x) * (1 - p.x) + p.y * p.y;
        float f = ((1 + p.x) * (1 + p.x) + p.y * p.y) / den;
        w = float2(0.5f * (f < 0.5f ? precise::log(f) : log_1p(4 * p.x / den)),
                   precise::atan2(2 * p.y, (1 - p.x) * (1 + p.x) - p.y * p.y));
        q = dt * float2(1 - p.x, p.y) / den;
    }
    dB = cmul(B, q);
}

// Compute the gradients gA, gB and gdt of one mode from the gradient gw of w = log dA and the
// gradient gb of dB. For a holomorphic y(x), the gradient of x is G_x = conj(dy/dx) G_y.
inline void discretize_grad(float dt, float2 A, float2 B, float2 gw, float2 gb, thread float2& gA,
                            thread float2& gB, thread float& gdt) {
    float2 a = dt * A;
    if (ZOH) {
        // E = expm1(a) / A, dE/dA = (dt exp(a) - E) / A and dE/d dt = exp(a)
        float den = dot(A, A), c = precise::cos(a.y), s = precise::sin(a.y / 2);
        float ex = precise::exp(a.x);
        float2 x = ex * float2(c, precise::sin(a.y));
        float2 E = cmulc(float2(exp_m1(a.x, ex) * c - 2 * s * s, x.y), A) / den;
        float2 h = cmul(B, cmulc(dt * x - E, A)) / den;  // B dE/dA
        gA = dt * gw + cmulc(gb, h);
        gB = cmulc(gb, E);
        gdt = dot(A, gw) + dot(cmul(B, x), gb);
    } else {
        // q = 1 / (1 - a/2), dB = dt B q, d dB/da = dt B q^2 / 2 and
        // d log dA / da = 1 / (1 - a^2/4) = v
        float2 q = float2(1 - a.x / 2, a.y / 2) / ((1 - a.x / 2) * (1 - a.x / 2) + a.y * a.y / 4);
        float2 m = float2(1 - (a.x * a.x - a.y * a.y) / 4, -(a.x * a.y) / 2);
        float2 v = float2(m.x, -m.y) / dot(m, m);
        float2 h = dt / 2 * cmul(B, cmul(q, q));
        float2 ga = cmulc(gw, v) + cmulc(gb, h);
        gA = dt * ga;
        gB = dt * cmulc(gb, q);
        gdt = dot(A, ga) + dot(cmul(B, q), gb);
    }
}
