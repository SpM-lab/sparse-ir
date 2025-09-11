#include "sparseir/poly.hpp"
#include "sparseir/impl/poly_impl.hpp"

namespace sparseir {

// PiecewiseLegendrePoly implementations
PiecewiseLegendrePoly::PiecewiseLegendrePoly(
    int polyorder, double xmin, double xmax, const Eigen::VectorXd &knots,
    const Eigen::VectorXd &delta_x, const Eigen::MatrixXd &data, int symm,
    int l, const Eigen::VectorXd &xm, const Eigen::VectorXd &inv_xs,
    const Eigen::VectorXd &norms)
    : polyorder(polyorder),
      xmin(xmin),
      xmax(xmax),
      knots(knots),
      delta_x(delta_x),
      data(data),
      symm(symm),
      l(l),
      xm(xm),
      inv_xs(inv_xs),
      norms(norms)
{

    if (!data.allFinite()) {
        throw std::invalid_argument("data contains NaN or Inf");
    }

    if (!std::is_sorted(knots.data(), knots.data() + knots.size())) {
        throw std::invalid_argument("knots must be monotonically increasing");
    }

    for (int i = 0; i < delta_x.size(); ++i) {
        if (std::abs(delta_x[i] - (knots[i + 1] - knots[i])) > 1e-10) {
            throw std::invalid_argument("delta_x must work with knots");
        }
    }
}

PiecewiseLegendrePoly::PiecewiseLegendrePoly(const Eigen::MatrixXd &data,
                                             const Eigen::VectorXd &knots,
                                             int l,
                                             const Eigen::VectorXd &delta_x_,
                                             int symm)
    : knots(knots), data(data), symm(symm), l(l)
{
    this->polyorder = data.rows();
    int nsegments = data.cols();

    if (knots.size() != nsegments + 1) {
        throw std::invalid_argument("Invalid knots array");
    }

    this->delta_x = delta_x_.size() > 0 ? delta_x_ : diff(knots);
    this->xm = Eigen::VectorXd::Zero(nsegments);
    for (int i = 0; i < nsegments; ++i) {
        this->xm[i] = 0.5 * (knots[i] + knots[i + 1]);
    }
    this->inv_xs = 2.0 / this->delta_x.array();
    this->norms = this->inv_xs.array().sqrt();

    this->xmin = knots(0);
    this->xmax = knots(knots.size() - 1);
}

double PiecewiseLegendrePoly::operator()(double x) const
{
    int i;
    double x_tilde;
    std::tie(i, x_tilde) = split(x);
    Eigen::VectorXd coeffs = data.col(i);
    std::vector<double> coeffs_vec(coeffs.data(),
                                   coeffs.data() + coeffs.size());
    double value = legval<double>(x_tilde, coeffs_vec) * norms(i);
    return value;
}

Eigen::VectorXd
PiecewiseLegendrePoly::operator()(const Eigen::VectorXd &xs) const
{
    Eigen::VectorXd results(xs.size());
    for (int idx = 0; idx < xs.size(); ++idx) {
        results[idx] = (*this)(xs[idx]);
    }
    return results;
}

double PiecewiseLegendrePoly::overlap(std::function<double(double)> f,
                                      const std::vector<double> &points) const
{
    double result = 0.0;

    std::vector<double> integration_points(knots.data(),
                                           knots.data() + knots.size());
    integration_points.insert(integration_points.end(), points.begin(),
                              points.end());
    std::sort(integration_points.begin(), integration_points.end());
    integration_points.erase(
        std::unique(integration_points.begin(), integration_points.end()),
        integration_points.end());

    for (size_t idx = 0; idx < integration_points.size() - 1; ++idx) {
        double a = integration_points[idx];
        double b = integration_points[idx + 1];

        auto integrand = [this, &f](double x) { return (*this)(x)*f(x); };
        double integral = gauss_legendre_quadrature(a, b, integrand);
        result += integral;
    }
    return result;
}

PiecewiseLegendrePoly PiecewiseLegendrePoly::deriv(int n) const
{
    Eigen::MatrixXd ddata = legder(data, n);

    for (int i = 0; i < ddata.cols(); ++i) {
        ddata.col(i) *= std::pow(inv_xs[i], n);
    }

    int new_symm = std::pow(-1, n) * symm;
    return PiecewiseLegendrePoly(ddata, *this, new_symm);
}

Eigen::VectorXd PiecewiseLegendrePoly::derivs(double x) const
{
    std::vector<double> res;
    res.push_back((*this)(x));
    PiecewiseLegendrePoly newppoly = *this;
    for (int i = 2; i <= polyorder; ++i) {
        newppoly = newppoly.deriv();
        res.push_back(newppoly(x));
    }
    return Eigen::Map<Eigen::VectorXd>(res.data(), res.size());
}

Eigen::VectorXd PiecewiseLegendrePoly::refine_grid(const Eigen::VectorXd &grid,
                                                   int alpha) const
{
    Eigen::VectorXd refined((grid.size() - 1) * alpha + 1);

    for (size_t i = 0; i < static_cast<size_t>(grid.size() - 1); ++i) {
        double start = grid[i];
        double step = (grid[i + 1] - grid[i]) / alpha;
        for (int j = 0; j < alpha; ++j) {
            refined[i * alpha + j] = start + j * step;
        }
    }
    refined[refined.size() - 1] = grid[grid.size() - 1];
    return refined;
}

double PiecewiseLegendrePoly::bisect(double a, double b, double fa,
                                     double eps_x) const
{
    while (true) {
        double mid = static_cast<double>(midpoint(a, b));
        if (closeenough(a, mid, eps_x)) {
            return mid;
        }

        double fmid = (*this)(mid);
        if (std::signbit(fa) != std::signbit(fmid)) {
            b = mid;
        } else {
            a = mid;
            fa = fmid;
        }
    }
}

Eigen::VectorXd PiecewiseLegendrePoly::roots() const
{
    Eigen::VectorXd grid = this->knots;

    Eigen::VectorXd refined_grid = refine_grid(grid, 2);
    std::function<double(double)> f = [this](double x) {
        return this->operator()(x);
    };
    std::vector<double> refined_grid_vec(
        refined_grid.data(), refined_grid.data() + refined_grid.size());
    std::vector<double> roots = find_all(f, refined_grid_vec);
    return Eigen::Map<Eigen::VectorXd>(roots.data(), roots.size());
}

std::pair<int, double> PiecewiseLegendrePoly::split(double x) const
{
    if (x < xmin || x > xmax) {
        throw std::domain_error("x is outside the domain");
    }

    auto it = std::lower_bound(knots.data(), knots.data() + knots.size(), x);
    int i = std::max(0, int(it - knots.data() - 1));
    i = std::min<int>(i, knots.size() - 2);

    double x_tilde = (x - xm[i]) * inv_xs[i];
    return std::make_pair(i, x_tilde);
}

double PiecewiseLegendrePoly::gauss_legendre_quadrature(
    double a, double b, std::function<double(double)> f) const
{
    static const double xg[] = {-0.906179845938664, -0.538469310105683, 0.0,
                                0.538469310105683, 0.906179845938664};
    static const double wg[] = {0.236926885056189, 0.478628670499366,
                                0.568888888888889, 0.478628670499366,
                                0.236926885056189};
    double c1 = (b - a) / 2.0;
    double c2 = (b + a) / 2.0;
    double integral = 0.0;
    for (int j = 0; j < 5; ++j) {
        double x = c1 * xg[j] + c2;
        integral += wg[j] * f(x);
    }
    integral *= c1;
    return integral;
}

std::complex<double> compute_unl_inner(const PiecewiseLegendrePoly &poly,
                                       int wn)
{
    double wred = M_PI / 4.0 * wn;
    Eigen::VectorXcd phase_wi = phase_stable(poly, wn);

    std::complex<double> res(0.0, 0.0);

    int num_orders = poly.get_data().rows();
    int num_j = poly.get_data().cols();

    for (int order = 0; order < num_orders; ++order) {
        int l = order;
        for (int j = 0; j < num_j; ++j) {
            double data_value = poly.get_data()(order, j);
            double delta_x_j = poly.delta_x(j);
            double norm_j = poly.norms(j);

            double wred_delta_x = wred * delta_x_j;
            std::complex<double> tnl = get_tnl(l, wred_delta_x);
            std::complex<double> phase = phase_wi(j);

            res += data_value * tnl * phase / norm_j;
        }
    }

    res /= std::sqrt(2.0);

    return res;
}

std::complex<double> get_tnl(int l, double w)
{
    double abs_w = std::abs(w);
    // Compute spherical Bessel function of order l at abs_w
    double sph_bessel = sphericalbesselj(l, abs_w);
    // Compute 2i^l
    std::complex<double> im_unit(0.0, 1.0);
    std::complex<double> im_power = std::pow(im_unit, l);
    std::complex<double> result = 2.0 * im_power * sph_bessel;

    if (w < 0.0) {
        return std::conj(result);
    } else {
        return result;
    }
}

std::pair<std::vector<double>, std::vector<int>>
shift_xmid(const std::vector<double> &knots, const std::vector<double> &delta_x)
{
    size_t N = delta_x.size();
    std::vector<double> delta_x_half(N);
    for (size_t i = 0; i < N; ++i) {
        delta_x_half[i] = delta_x[i] / 2.0;
    }

    // Compute xmid_m1
    std::vector<double> xmid_m1(N);
    std::vector<double> cumsum(N);
    cumsum[0] = delta_x[0];
    for (size_t i = 1; i < N; ++i) {
        cumsum[i] = cumsum[i - 1] + delta_x[i];
    }
    for (size_t i = 0; i < N; ++i) {
        xmid_m1[i] = cumsum[i] - delta_x_half[i];
    }

    // Compute xmid_p1
    std::vector<double> xmid_p1(N);
    std::vector<double> rev_delta_x(N);
    for (size_t i = 0; i < N; ++i) {
        rev_delta_x[N - 1 - i] = delta_x[i];
    }
    std::vector<double> rev_cumsum(N);
    rev_cumsum[0] = rev_delta_x[0];
    for (size_t i = 1; i < N; ++i) {
        rev_cumsum[i] = rev_cumsum[i - 1] + rev_delta_x[i];
    }
    for (size_t i = 0; i < N; ++i) {
        xmid_p1[i] = -rev_cumsum[N - 1 - i] + delta_x_half[i];
    }

    // Compute xmid_0
    std::vector<double> xmid_0(N);
    for (size_t i = 0; i < N; ++i) {
        xmid_0[i] =
            knots[i + 1] - delta_x_half[i]; // Assuming knots has length N + 1
    }

    // Compute shift
    std::vector<int> shift(N);
    for (size_t i = 0; i < N; ++i) {
        shift[i] = static_cast<int>(std::round(xmid_0[i]));
    }

    // Compute diff
    std::vector<double> diff(N);
    for (size_t i = 0; i < N; ++i) {
        int idx = shift[i] + 1; // shift can be -1, 0, 1; idx ranges from 0 to 2
        if (idx == 0) {
            diff[i] = xmid_m1[i];
        } else if (idx == 1) {
            diff[i] = xmid_0[i];
        } else if (idx == 2) {
            diff[i] = xmid_p1[i];
        } else {
            // Should not happen
            throw std::runtime_error("Invalid shift value");
        }
    }

    return std::make_pair(diff, shift);
}

Eigen::VectorXcd phase_stable(const PiecewiseLegendrePoly &poly, int wn)
{
    const std::vector<double> knots(poly.knots.data(),
                                    poly.knots.data() + poly.knots.size());
    const std::vector<double> delta_x(
        poly.delta_x.data(), poly.delta_x.data() + poly.delta_x.size());

    // Compute xmid_diff and extra_shift
    std::pair<std::vector<double>, std::vector<int>> shift_result =
        shift_xmid(knots, delta_x);
    const std::vector<double> &xmid_diff = shift_result.first;
    const std::vector<int> &extra_shift = shift_result.second;

    size_t N = delta_x.size();
    Eigen::VectorXcd phase_wi(N);

    for (size_t i = 0; i < N; ++i) {
        int exponent = wn * (extra_shift[i] + 1);
        int exponent_mod4 = ((exponent % 4) + 4) % 4; // Ensure positive modulo

        std::complex<double> im_power;
        switch (exponent_mod4) {
        case 0:
            im_power = std::complex<double>(1.0, 0.0);
            break;
        case 1:
            im_power = std::complex<double>(0.0, 1.0);
            break;
        case 2:
            im_power = std::complex<double>(-1.0, 0.0);
            break;
        case 3:
            im_power = std::complex<double>(0.0, -1.0);
            break;
        }

        double arg = M_PI * wn * xmid_diff[i] / 2.0;
        std::complex<double> cispi = std::polar(1.0, arg); // exp(i * arg)

        phase_wi(i) = im_power * cispi;
    }

    return phase_wi;
}

// Explicit template instantiations
template class PiecewiseLegendreFT<Bosonic>;
template class PiecewiseLegendreFT<Fermionic>;
template class PiecewiseLegendreFTVector<Bosonic>;
template class PiecewiseLegendreFTVector<Fermionic>;

// Explicit instantiations for find_extrema and related templates
template std::vector<MatsubaraFreq<Fermionic>>
find_extrema(const PiecewiseLegendreFT<Fermionic> &u_hat, bool positive_only);
template std::vector<MatsubaraFreq<Bosonic>>
find_extrema(const PiecewiseLegendreFT<Bosonic> &u_hat, bool positive_only);

// Explicit instantiations for power_model
// template PowerModel<double> PiecewiseLegendreFT<Bosonic>::power_model(const
// Bosonic& stat, const PiecewiseLegendrePoly& poly); template
// PowerModel<double> PiecewiseLegendreFT<Fermionic>::power_model(const
// Fermionic& stat, const PiecewiseLegendrePoly& poly);

std::complex<double> evalpoly(std::complex<double> x,
                              const std::vector<double> &coeffs)
{
    std::complex<double> result(0, 0);
    std::complex<double> xn(1, 0);
    for (double c : coeffs) {
        result += c * xn;
        xn *= x;
    }
    return result;
}

void symmetrize_matsubara_inplace(std::vector<int> &xs)
{
    // is sorted
    if (!std::is_sorted(xs.begin(), xs.end())) {
        throw std::runtime_error("points must be sorted");
    }
    if (xs.empty())
        return;
    if (xs.front() < 0)
        throw std::runtime_error("points must be non-negative");

    std::vector<int> neg(xs.rbegin(), xs.rend());
    for (auto &x : neg) {
        x = -x;
    }
    if (std::abs(xs.front()) < 1e-12 && !xs.empty()) {
        xs.erase(xs.begin());
    }
    xs.insert(xs.begin(), neg.begin(), neg.end());
}
template std::function<double(int)>
func_for_part(const PiecewiseLegendreFT<Fermionic> &polyFT,
              std::function<double(std::complex<double>)> part);

template std::function<double(int)>
func_for_part(const PiecewiseLegendreFT<Bosonic> &polyFT,
              std::function<double(std::complex<double>)> part);

} // namespace sparseir