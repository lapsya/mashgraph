#include "align.h"
#include <string>
#include <algorithm>
#include <math.h>
#include <set>
#include <vector>

using std::string;
using std::cout;
using std::endl;
using std::get;
using std::make_tuple;
using std::tuple;
using std::tie;
using std::min;
using std::max;
using std::set;
using std::vector;
using std::sort;

// Align two images using MSE by their R channels
// The resuting image is a cropped copy of `img1` and has `overlay`
// channel set to corresponding value from `img2`
// using cross-correlation if `cross` is not 0 and MSE otherwise
Image img_shift(const Image img1, const Image img2, char overlay, char cross)
{
    double mse_min = -1.0;
    double cross_max = 0.0;
    int radius = 15;
    int best_row_shift = -radius, best_col_shift = -radius;

    // find best shift by shifting img2
    // and finding the metrics minimum
    for (int row_shift = -radius; row_shift <= radius; row_shift++) {
        for (int col_shift = -radius; col_shift <= radius; col_shift++) {
            double mse = 0.0;
            double cross_sum = 0.0;

            // calculate borders of intersection of shifted images
            uint img1_row_first = max(0, row_shift);
            uint img1_row_last = min(img2.n_rows + row_shift, img1.n_rows);
            uint img1_col_first = max(0, col_shift);
            uint img1_col_last = min(img2.n_cols + col_shift, img1.n_cols);

            // calculate metrics
            for (uint x = img1_row_first; x < img1_row_last; x++) {
                for (uint y = img1_col_first; y < img1_col_last; y++) {
                    uint img2_r = get<0>(img2(x - row_shift, y - col_shift));
                    uint img1_r = get<0>(img1(x, y));
                    if (cross) {
                        cross_sum += img2_r * img1_r;
                    } else {
                        mse += (int(img2_r) - img1_r) * (int(img2_r) - img1_r);
                    }
                }
            }

            // compare with the optimal metrics value
            if (cross) {
                cross_sum /= (img1_col_last - img1_col_first) * (img1_row_last - img1_row_first);
                if (cross_sum > cross_max) {
                    cross_max = cross_sum;
                    best_row_shift = row_shift;
                    best_col_shift = col_shift;
                }
            } else {
                mse /= (img1_col_last - img1_col_first) * (img1_row_last - img1_row_first);
                if (mse_min < 0.0 || mse < mse_min) {
                    mse_min = mse;
                    best_row_shift = row_shift;
                    best_col_shift = col_shift;
                }
            }
        }
    }
    //cout << "mse_min: " << mse_min << endl;
    //cout << "cross_max: " << cross_max << endl;
    //cout << "best row shift: " << best_row_shift << ", best col shift: " << best_col_shift << endl;

    // calculate borders of the resulting overlaid image
    uint img1_row_first = max(0, best_row_shift);
    uint img1_row_last = min(img2.n_rows + best_row_shift, img1.n_rows);
    uint img1_col_first = max(0, best_col_shift);
    uint img1_col_last = min(img2.n_cols + best_col_shift, img1.n_cols);

    // overlay images
    Image res_img = img1.submatrix(img1_row_first, img1_col_first, img1_row_last - img1_row_first, img1_col_last - img1_col_first);
    for (uint x = img1_row_first; x < img1_row_last; x++) {
        for (uint y = img1_col_first; y < img1_col_last; y++) {
            if (overlay == 'g') {
                res_img(x - img1_row_first, y - img1_col_first) = make_tuple(get<0>(img1(x, y)), get<1>(img2(x - best_row_shift, y - best_col_shift)), get<2>(img1(x, y)));
            } else {
                res_img(x - img1_row_first, y - img1_col_first) = make_tuple(get<0>(img2(x - best_row_shift, y - best_col_shift)), get<1>(img1(x, y)), get<2>(img1(x, y)));
            }
        }
    }
    return res_img;
}

Image border_cut(Image src_image);

Image align(Image srcImage, bool isPostprocessing, std::string postprocessingType, double fraction, bool isMirror,
            bool isInterp, bool isSubpixel, double subScale)
{
    // cut original image into 3 images according to channels
    uint img_height = srcImage.n_rows / 3;
    Image img_b_uncut = srcImage.submatrix(0, 0, img_height, srcImage.n_cols);
    Image img_g_uncut = srcImage.submatrix(img_height, 0, img_height, srcImage.n_cols);
    Image img_r_uncut = srcImage.submatrix(2 * img_height, 0, img_height, srcImage.n_cols);

    // cut borders off each image using Canny algorithm
    Image img_b_cut = border_cut(img_b_uncut);
    Image img_g_cut = border_cut(img_g_uncut);
    Image img_r_cut = border_cut(img_r_uncut);

    // call `img_shift` to overlay images using MSE metrics
    // which showed better results than cross-correlation
    Image img_bg = img_shift(img_b_cut, img_g_cut, 'g', 0);
    Image img = img_shift(img_bg, img_r_cut, 'r', 0);
    return img;
}

class Filter
{
public:
    Matrix<double> kernel;
    int radius;
    bool is_sobel;
    Filter(Matrix<double> &k, bool sobel=false): kernel(k), radius(k.n_rows / 2), is_sobel(sobel) {}
    tuple<uint, uint, uint> operator () (const Image &m) const
    {
        uint size = 2 * radius + 1;
        uint r, g, b;
        double red = 0.0, green = 0.0, blue = 0.0;
        for (uint i = 0; i < size; i++) {
            for (uint j = 0; j < size; j++) {
                tie(r, g, b) = m(i, j);
                red += kernel(i, j) * r;
                green += kernel(i, j) * g;
                blue += kernel(i, j) * b;
            }
        }
        if (!is_sobel) {
            red = red < 0 ? 0 : red > 255 ? 255 : red;
            green = green < 0 ? 0 : green > 255 ? 255 : green;
            blue = blue < 0 ? 0 : blue > 255 ? 255 : blue;
        }
        return make_tuple(uint(red), uint(green), uint(blue));
    }
};


Image sobel_x(Image src_image) {
    Matrix<double> kernel = {{-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1}};
    return src_image.unary_map(Filter(kernel, true));
}

Image sobel_y(Image src_image) {
    Matrix<double> kernel = {{ 1,  2,  1},
                             { 0,  0,  0},
                             {-1, -2, -1}};
    return src_image.unary_map(Filter(kernel, true));
}

Image unsharp(Image src_image) {
    Matrix<double> sharp_matrix = {{-1.0/6, -2.0/3, -1.0/6},
                                   {-2.0/3, 13.0/3, -2.0/3},
                                   {-1.0/6, -2.0/3, -1.0/6}};
    for (uint i = 0; i < 3; i++) {
        for (uint j = 0; j < 3; j++) {
            cout << sharp_matrix(i, j) << " ";
        }
        cout << endl;
    }
    return custom(src_image, sharp_matrix);
}

Image gray_world(Image src_image) {
    double s_r = 0.0, s_g = 0.0, s_b = 0.0;
    uint r, g, b;
    for (uint i = 0; i < src_image.n_rows; i++) {
        for (uint j = 0; j < src_image.n_cols; j++) {
            tie(r, g, b) = src_image(i, j);
            s_r += r;
            s_g += g;
            s_b += b;
        }
    }
    s_r /= src_image.n_rows * src_image.n_cols;
    s_g /= src_image.n_rows * src_image.n_cols;
    s_b /= src_image.n_rows * src_image.n_cols;
    double s = (s_r + s_g + s_b) / 3;
    for (uint i = 0; i < src_image.n_rows; i++) {
        for (uint j = 0; j < src_image.n_cols; j++) {
            tie(r, g, b) = src_image(i, j);
            uint new_r = r * (s / s_r) > 255 ? 255 : r * (s / s_r);
            uint new_g = g * (s / s_g) > 255 ? 255 : g * (s / s_g);
            uint new_b = b * (s / s_b) > 255 ? 255 : b * (s / s_b);
            src_image(i, j) = make_tuple(new_r, new_g, new_b);
        }
    }
    return src_image;
}

Image resize(Image src_image, double scale) {
    return src_image;
}


Image custom(Image src_image, Matrix<double> kernel) {
    // Function custom is useful for making concrete linear filtrations
    // like gaussian or sobel. So, we assume that you implement custom
    // and then implement other filtrations using this function.
    // sobel_x and sobel_y are given as an example.
    return src_image.unary_map(Filter(kernel));
}

Image autocontrast(Image src_image, double fraction) {
    int histogram[256] = {0};

    for (uint i = 0; i < src_image.n_rows; i++) {
        for (uint j = 0; j < src_image.n_cols; j++) {
            uint r, g, b;
            tie(r, g, b) = src_image(i, j);
            uint y = 0.2125 * r + 0.7154 * g + 0.0721 * b;
            y = y > 255 ? 255 : y;
            histogram[y] += 1;
        }
    }
    int y_min = 0, y_max = 255;
    int to_cut = fraction * src_image.n_rows * src_image.n_cols / 2;
    int pixel_count = 0;
    for (; pixel_count < to_cut; y_min++) {
        pixel_count += histogram[y_min];
    }
    y_min--;
    pixel_count = 0;
    for (; pixel_count < to_cut; y_max--) {
        pixel_count += histogram[y_max];
    }
    y_max++;
    for (uint i = 0; i < src_image.n_rows; i++) {
        for (uint j = 0; j < src_image.n_cols; j++) {
            uint r, g, b;
            tie(r, g, b) = src_image(i, j);
            int new_r = ((int(r) - y_min) * 255) / (y_max - y_min);
            int new_g = ((int(g) - y_min) * 255) / (y_max - y_min);
            int new_b = ((int(b) - y_min) * 255) / (y_max - y_min);
            new_r = new_r > 255 ? 255 : (new_r < 0 ? 0 : new_r);
            new_g = new_g > 255 ? 255 : (new_g < 0 ? 0 : new_g);
            new_b = new_b > 255 ? 255 : (new_b < 0 ? 0 : new_b);
            src_image(i, j) = make_tuple(new_r, new_g, new_b);
        }
    }

    return src_image;
}

Image gaussian(Image src_image, double sigma, int radius)  {

    // calculate kernel matrix
    int size = 2 * radius + 1;
    Matrix<double> kernel(size, size);
    double kernel_sum = 0.0;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <=radius; j++) {
            kernel(i + radius, j + radius) = exp((-i * i - j * j) / (2.0 * sigma * sigma)) / (2.0 * M_PI * sigma * sigma);
            kernel_sum += kernel(i + radius, j + radius);
        }
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel(i, j) = kernel(i, j) / kernel_sum;
        }
    }

    // call `custom` that will apply filtration
    return custom(src_image, kernel);
}

Image gaussian_separable(Image src_image, double sigma, int radius) {

    // create "kernel" vector
    int size = 2 * radius + 1;
    Matrix<double> kernel(1, size);
    double kernel_sum = 0.0;
    for (int i = -radius; i <= radius; i++) {
        kernel(0, i + radius) = exp((-i) * i / (2.0 * sigma * sigma)) / (sqrt(2.0 * M_PI) * sigma);
        kernel_sum += kernel(0, i + radius);
    }
    for (int i = 0; i <= 2 * radius; i++) {
        kernel(0, i) = kernel(0, i) / kernel_sum;
    }

    // apply filtration to rows
    Image after_rows(src_image.n_rows, src_image.n_cols);
    const auto start_i = radius;
    const auto end_i = src_image.n_rows - radius;
    const auto start_j = radius;
    const auto end_j = src_image.n_cols - radius;
    for (uint i = start_i; i < end_i; ++i) {
        for (uint j = 0; j < after_rows.n_cols; ++j) {
            double new_r = 0.0, new_g = 0.0, new_b = 0.0;
            uint r, g, b;
            for (int n_rows = -radius; n_rows <= radius; n_rows++) {
                tie(r, g, b) = src_image(i + n_rows, j);
                new_r += r * kernel(0, n_rows + radius);
                new_g += g * kernel(0, n_rows + radius);
                new_b += b * kernel(0, n_rows + radius);
            }
            after_rows(i, j) = make_tuple(int(new_r), int(new_g), int(new_b));
        }
    }

    // apply filtration to cols
    Image after_cols(src_image.n_rows, src_image.n_cols);
    for (uint j = start_j; j < end_j; ++j) {
        for (uint i = start_i; i < end_i; ++i) {
            double new_r = 0.0, new_g = 0.0, new_b = 0.0;
            uint r, g, b;
            for (int n_cols = -radius; n_cols <= radius; n_cols++) {
                tie(r, g, b) = after_rows(i, j + n_cols);
                new_r += r * kernel(0, n_cols + radius);
                new_g += g * kernel(0, n_cols + radius);
                new_b += b * kernel(0, n_cols + radius);
            }
            after_cols(i, j) = make_tuple(int(new_r), int(new_g), int(new_b));
        }
    }

    return after_cols;
}

Image median(Image src_image, int radius) {
    uint start_i = radius;
    uint start_j = radius;
    uint end_i = src_image.n_rows - radius;
    uint end_j = src_image.n_cols - radius;
    Image filtered(src_image.n_rows, src_image.n_cols);
    for (uint i = start_i; i < end_i; i++) {
        for (uint j = start_j; j < end_j; j++) {
            vector<uint> reds, greens, blues;
            uint r, g, b;
            for (uint n_row = i - radius; n_row <= i + radius; n_row++) {
                for (uint n_col = j - radius; n_col <= j + radius; n_col++) {
                    tie(r, g, b) = src_image(n_row, n_col);
                    reds.push_back(r);
                    greens.push_back(g);
                    blues.push_back(b);
                }
            }
            sort(reds.begin(), reds.end());
            sort(greens.begin(), greens.end());
            sort(blues.begin(), blues.end());
            filtered(i, j) = make_tuple(reds[reds.size() / 2], greens[greens.size() / 2], blues[blues.size() / 2]);
        }
    }
    return filtered;
}

Image median_linear(Image src_image, int radius) {

    Image filtered(src_image.n_rows, src_image.n_cols);
    vector<uint> reds, greens, blues;
    uint r, g, b;

    for (uint i = radius; i < src_image.n_rows - radius; i++) {
        for (int j = 0; j < 2 * radius + 1; j++) {
            for (uint row = i - radius; row <= i + radius; row++) {
                tie(r, g, b) = src_image(i, j);
                reds.push_back(r);
                greens.push_back(g);
                blues.push_back(b);
            }
        }
        for (uint j = radius; j < src_image.n_cols - radius; j++) {
            vector<uint> sort_r(reds), sort_g(greens), sort_b(blues);
            sort(sort_r.begin(), sort_r.end());
            sort(sort_g.begin(), sort_g.end());
            sort(sort_b.begin(), sort_b.end());

            filtered(i, j) = make_tuple(sort_r[sort_r.size() / 2], sort_g[sort_g.size() / 2], sort_b[sort_b.size() / 2]);
            if (j + radius + 1 >= src_image.n_cols) {
                break;
            }
            for (uint row = i - radius; row <= i + radius; row++) {
                tie(r, g, b) = src_image(row, j + radius + 1);
                reds.erase(reds.begin());
                greens.erase(greens.begin());
                blues.erase(blues.begin());
                reds.push_back(r);
                greens.push_back(g);
                blues.push_back(b);
            }
        }
        reds.clear();
        greens.clear();
        blues.clear();
    }

    return filtered;
}

Image median_const(Image src_image, int radius) {
    return src_image;
}

Image canny(Image src_image, int threshold1, int threshold2) {

    // apply Gaussian filter
    Image blurred = gaussian(src_image, 1.4, 2);

    // apply Sobel filtration
    auto I_x = sobel_x(blurred);
    auto I_y = sobel_y(blurred);

    // calculate the matrix of gradients for each pixel
    Matrix<double> gradients(src_image.n_rows, src_image.n_cols);
    for (uint i = 0; i < gradients.n_rows; i++) {
        for (uint j = 0; j < gradients.n_cols; j++) {
            gradients(i, j) = sqrt(int(get<0>(I_x(i, j))) * int(get<0>(I_x(i, j))) + int(get<0>(I_y(i, j))) * int(get<0>(I_y(i, j))));
        }
    }

    // suppress non-maximums for each pixel
    // by comparing gradients with neighbours' gradients
    for (uint i = 0; i < gradients.n_rows; i++) {
        for (uint j = 0; j < gradients.n_cols; j++) {

            // figure out the neighbours we'll need
            int neighbour1 = floor((atan2(int(get<0>(I_y(i, j))), int(get<0>(I_x(i, j))))) / (M_PI / 4.0));
//            3  2  1
//           -4     0
//           -3 -2 -1
            if (neighbour1 == 4) {
                neighbour1 = -4;
            }
            int neighbour2 = neighbour1 >= 0 ? neighbour1 - 4 : neighbour1 + 4;
            int row_shift = 0, col_shift = 0;
            switch (max(neighbour1, neighbour2)) {
                case 0: {
                    row_shift = 0;
                    col_shift = 1;
                    break;
                }
                case 1: {
                    row_shift = -1;
                    col_shift = 1;
                    break;
                }
                case 2: {
                    row_shift = -1;
                    col_shift = 0;
                    break;
                }
                case 3: {
                    row_shift = -1;
                    col_shift = -1;
                    break;
                }
                default: {
                    break;
                }
            }

            // check for maximum with the first neighbour
            if (int(i) + row_shift >= 0 && int(i) + row_shift < int(gradients.n_rows)) {
                if (int(j) + col_shift >= 0 && int(j) + col_shift < int(gradients.n_cols)) {
                    if (gradients(i, j) <= gradients(i + row_shift, j + col_shift)) {
                        gradients(i, j) = 0;
                    }
                }
            }
            // check for maximum with the second neighbour
            if (int(i) - row_shift >= 0 && int(i) - row_shift < int(gradients.n_rows)) {
                if (int(j) - col_shift >= 0 && int(j) - col_shift < int(gradients.n_cols)) {
                    if (gradients(i, j) <= gradients(i - row_shift, j - col_shift)) {
                        gradients(i, j) = 0;
                    }
                }
            }
            // set gradients less than the first threshold to `0`s
            if (gradients(i, j) < threshold1) {
                gradients(i, j) = 0.0;
            }
        }
    }

    // hysteresis border tracking algorithm


    int n_connections = 0;

    // connections containing at least one
    // strong gradient are considered valid
    set<int> valid_connections;

    // connections are divided into several equivalency classes
    // stored in this structure
    vector<set<int>> equivalency_classes;

    Matrix<int> labels(blurred.n_rows, blurred.n_cols);
    for (uint i = 0; i < labels.n_rows; i++) {
        for (uint j = 0; j < labels.n_cols; j++) {
            labels(i, j) = 0;
        }
    }

    // label all the pixels with a number
    // of the connection they belong to
    for (uint i = 0; i < gradients.n_rows; i++) {
        for (uint j = 0; j < gradients.n_cols; j++) {
            if (gradients(i, j) < 0.01) {
                continue;
            }
            set<int> connected_to;

            // find labels of all neighbours and add them to `connected_to`
            for (int neighbour_row = int(i) - 1; neighbour_row <= int(i) + 1; neighbour_row++) {
                for (int neighbour_col = int(j) - 1; neighbour_col <= int(j) + 1; neighbour_col++) {

                    // check that such neighbour exists and add its label to `connected_to`
                    if (neighbour_row != int(i) || neighbour_col != int(j)) {
                        if (neighbour_row >= 0 && neighbour_row < int(gradients.n_rows)) {
                            if (neighbour_col >= 0 && neighbour_col < int(gradients.n_cols)) {
                                if (gradients(neighbour_row, neighbour_col) > 0) {
                                    if (labels(neighbour_row, neighbour_col) > 0) {
                                        connected_to.insert(labels(neighbour_row, neighbour_col));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (connected_to.empty()) {
                // create a new connection
                labels(i, j) = ++n_connections;
            } else if (connected_to.size() == 1) {
                // label as a part of existing connection
                labels(i, j) = *connected_to.begin();
            } else {
                // label as a part of existing connection
                // and record that some connections are equivalent
                labels(i, j) = *connected_to.begin();

                // if there are no equivalency classes yet, create one
                if (equivalency_classes.empty()) {
                    set<int> equiv_class;
                    equiv_class.insert(*connected_to.begin());
                    equivalency_classes.push_back(equiv_class);
                }

                // find the corresponding equivalency class
                // and add numbers of neighbouring connections to it
                bool equiv_class_exists = false;
                for (auto equiv_class = equivalency_classes.begin(); equiv_class != equivalency_classes.end(); equiv_class++) {
                    for (auto neighbour_conn = connected_to.begin(); neighbour_conn != connected_to.end(); neighbour_conn++) {
                        if ((*equiv_class).find(*neighbour_conn) != (*equiv_class).end()) {
                            equiv_class_exists = true;
                            for (auto k = connected_to.begin(); k != connected_to.end(); k++) {
                                (*equiv_class).insert(*k);
                            }
                            break;
                        }
                    }
                }

                // if connection is not in any of the
                // equivalency classes yet, create one
                if (!equiv_class_exists) {
                    set<int> equiv_class;
                    equiv_class.insert(*connected_to.begin());
                    equivalency_classes.push_back(equiv_class);
                }
            }

            // record if the connection is valid
            if (gradients(i, j) >= threshold2) {
                valid_connections.insert(labels(i, j));
            }
        }
    }

    // assign `0`s to gradients of pixels in the non-valid
    // and not equivalent to valid connections
    for (uint i = 0; i < gradients.n_rows; i++) {
        for (uint j = 0; j < gradients.n_cols; j++) {
            bool is_valid = false;

            if (labels(i, j) > 0) {
                if (valid_connections.find(labels(i, j)) != valid_connections.end()) {
                    is_valid = true;
                } else {

                    // find equivalency class connection belongs to
                    for (auto equiv_class = equivalency_classes.begin(); equiv_class != equivalency_classes.end(); equiv_class++) {
                        if ((*equiv_class).find(labels(i, j)) != (*equiv_class).end()) {

                            // find out if there's at least one
                            // valid connection in equivalency class
                            for (auto connection = (*equiv_class).begin(); connection != (*equiv_class).end(); connection++) {
                                if (valid_connections.find(*connection) != valid_connections.end()) {
                                    is_valid = true;
                                }
                            }
                        }
                    }
                }
            }

            if (!is_valid) {
                gradients(i, j) = 0.0;
            } else {
                gradients(i, j) = 255;
            }
        }
    }

    // create and return the border map
    Image grad(gradients.n_rows, gradients.n_cols);
    for (uint i = 0; i < gradients.n_rows; i++) {
        for (uint j = 0; j < gradients.n_cols; j++) {
            grad(i, j) = make_tuple(gradients(i, j), gradients(i, j), gradients(i, j));
        }
    }
    return grad;
}

Image border_cut(Image src_image)
{
    Image border_map = canny(src_image, 10, 100);
    int suppression_rad = 3;

    // find upper border
    int row_search_limit = 0.05 * src_image.n_rows;

    // find first maximum
    int max1_row = 0;
    int max1_n = 0;
    for (int i = 0; i < row_search_limit; i++) {
        int border_sum = 0;
        for (uint j = 0; j < src_image.n_cols; j++) {
            border_sum += get<0>(border_map(i, j));
        }
        if (border_sum >= max1_n) {
            max1_n = border_sum;
            max1_row = i;
        }
    }

    // set nearby rows to `0`s
    for (int i = -suppression_rad; i <= suppression_rad; i++) {
        if (i && max1_row + i >= 0) {
            for (uint j = 0; j < src_image.n_cols; j++) {
                border_map(max1_row + i, j) = make_tuple(0, 0, 0);
            }
        }
    }

    // find second maximum
    int max2_row = 0;
    int max2_n = 0;
    for (int i = 0; i < row_search_limit; i++) {
        if (i == max1_row) {
            continue;
        }
        int border_sum = 0;
        for (uint j = 0; j < src_image.n_cols; j++) {
            border_sum += get<0>(border_map(i, j));
        }
        if (border_sum >= max2_n) {
            max2_n = border_sum;
            max2_row = i;
        }
    }

    // find bottom border

    // find first maximum
    int max1_row_bot = src_image.n_rows - 1;
    int max1_n_bot = 0;
    for (int i = src_image.n_rows - 1; i > int(src_image.n_rows) - 1 - row_search_limit; i--) {
        int border_sum = 0;
        for (uint j = 0; j < src_image.n_cols; j++) {
            border_sum += get<0>(border_map(i, j));
        }
        if (border_sum >= max1_n_bot) {
            max1_n_bot = border_sum;
            max1_row_bot = i;
        }
    }

    // set nearby rows to `0`s
    for (int i = -suppression_rad; i <= suppression_rad; i++) {
        if (i && max1_row_bot + i < int(src_image.n_rows)) {
            for (uint j = 0; j < src_image.n_cols; j++) {
                border_map(max1_row_bot + i, j) = make_tuple(0, 0, 0);
            }
        }
    }

    // find second maximum
    int max2_row_bot = src_image.n_rows - 1;
    int max2_n_bot = 0;
    for (int i = src_image.n_rows - 1; i > int(src_image.n_rows) - 1 - row_search_limit; i--) {
        if (i == max1_row_bot) {
            continue;
        }
        int border_sum = 0;
        for (uint j = 0; j < src_image.n_cols; j++) {
            border_sum += get<0>(border_map(i, j));
        }
        if (border_sum >= max2_n_bot) {
            max2_n_bot = border_sum;
            max2_row_bot = i;
        }
    }

    // find left border

    int col_search_limit = 0.07 * src_image.n_cols;

    // find first maximum
    int max1_col = 0;
    int max1_n_left = 0;
    for (int j = 3; j < col_search_limit; j++) {
        int border_sum = 0;
        for (uint i = 0; i < src_image.n_rows; i++) {
            border_sum += get<0>(border_map(i, j));
        }
        if (border_sum >= max1_n_left) {
            max1_n_left = border_sum;
            max1_col = j;
        }
    }

    // set nearby cols to `0`s
    for (int j = -suppression_rad; j <= suppression_rad; j++) {
        if (j && max1_col + j >= 0) {
            for (uint i = 0; i < src_image.n_rows; i++) {
                border_map(i, max1_col + j) = make_tuple(0, 0, 0);
            }
        }
    }

    // find second maximum
    int max2_col = 0;
    int max2_n_left = 0;
    for (int j = 3; j < col_search_limit; j++) {
        if (j == max1_col) {
            continue;
        }
        int border_sum = 0;
        for (uint i = 0; i < src_image.n_rows; i++) {
            border_sum += get<0>(border_map(i, j));
        }
        if (border_sum >= max2_n_left) {
            max2_n_left = border_sum;
            max2_col = j;
        }
    }

    // find right border

    // find first maximum
    int max1_col_right = src_image.n_cols - 1;
    int max1_n_right = 0;
    for (int j = src_image.n_cols - 3; j > int(src_image.n_cols) - 1 - col_search_limit; j--) {
        int border_sum = 0;
        for (uint i = 0; i < src_image.n_rows; i++) {
            border_sum += get<0>(border_map(i, j));
        }
        if (border_sum >= max1_n_right) {
            max1_n_right = border_sum;
            max1_col_right = j;
        }
    }

    // set nearby cols to `0`s
    for (int j = -suppression_rad; j <= suppression_rad; j++) {
        if (j && max1_col_right + j < int(src_image.n_cols)) {
            for (uint i = 0; i < src_image.n_rows; i++) {
                border_map(i, max1_col_right + j) = make_tuple(0, 0, 0);
            }
        }
    }


    // find second maximum
    int max2_col_right = src_image.n_cols - 1;
    int max2_n_right = 0;
    for (int j = src_image.n_cols - 3; j < int(src_image.n_cols) - 1 - col_search_limit; j--) {
        if (j == max1_col_right) {
            continue;
        }
        int border_sum = 0;
        for (uint i = 0; i < src_image.n_rows; i++) {
            border_sum += get<0>(border_map(i, j));
        }
        if (border_sum >= max2_n_right) {
            max2_n_right = border_sum;
            max2_col_right = j;
        }
    }

    int upper_row_cut = max(max1_row, max2_row);
    int bottom_row_cut = min(max1_row_bot, max2_row_bot);
    int left_col_cut = max(max1_col, max2_col);
    int right_col_cut = min(max1_col_right, max2_col_right);
    return src_image.submatrix(upper_row_cut, left_col_cut, bottom_row_cut - upper_row_cut, right_col_cut - left_col_cut);
}
