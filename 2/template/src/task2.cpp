#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::tuple;
using std::tie;
using std::make_tuple;
using std::get;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;

    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);

    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels,
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

// Unary operator for filtering with Sobel
template <typename ValueT>
class Filter
{
public:
    Matrix<ValueT> kernel;

    // Excessive variables requested by unary_map
    int vert_radius, hor_radius, radius;

    Filter(Matrix<ValueT> &k): kernel(k), vert_radius(k.n_rows / 2), hor_radius(k.n_rows / 2), radius(k.n_rows / 2) {}

    ValueT operator () (const Matrix<ValueT> &neighbourhood) const
    {
        uint size = 2 * radius + 1;
        ValueT new_value{};
        for (uint i = 0; i < size; i++) {
            for (uint j = 0; j < size; j++) {
                new_value += neighbourhood(i, j) * kernel(i, j);
            }
        }
        return new_value;
    }
};

// Unary operator for local binary pattern
class LBP
{
public:
    const int radius = 1;
    const int vert_radius = 1;
    const int hor_radius = 1;
    LBP() {}
    float operator () (const Matrix<float> &neighbourhood) const
    {
        int size = 2 * radius + 1;
        float lbp_value = 0;
        float center = neighbourhood(radius, radius);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (i == radius && j == radius) {
                    continue;
                }
                lbp_value *= 2;
                lbp_value += neighbourhood(i, j) > center ? 1 : 0;
            }
        }
        return lbp_value;
    }
};


// Extract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    // pre-define constants for histograms
    const int n_blocks = 256;
    const int n_segments = 32;
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        auto image = data_set[image_idx].first;

        // turn image into brightness matrix
        Matrix<float> image_matrix(image->TellHeight(), image->TellWidth());
        for (int i = 0; i < image->TellHeight(); ++i) {
            for (int j = 0; j < image->TellWidth(); ++j) {
                RGBApixel pixel = image->GetPixel(j, i);
                int s = pixel.Red + pixel.Green + pixel.Blue;
                image_matrix(i, j) = s;
            }
        }

        // initialize Sobel kernels
        Matrix<float> kernel_hor = {{-1, 0, 1},
                                    {-2, 0, 2},
                                    {-1, 0, 1}};
        Matrix<float> kernel_ver = {{ 1, 2, 1},
                                    { 0, 0, 0},
                                    {-1,-2,-1}};

        // perform convolution
        Matrix<float> hor_Sobel = image_matrix.unary_map(Filter<float>(kernel_hor));
        Matrix<float> ver_Sobel = image_matrix.unary_map(Filter<float>(kernel_ver));
        Matrix<float> grad_abs(hor_Sobel.n_rows, hor_Sobel.n_cols);
        Matrix<float> grad_angle(hor_Sobel.n_rows, hor_Sobel.n_cols);

        // compute gradients
        for (uint i = 0; i < hor_Sobel.n_rows; ++i) {
            for (uint j = 0; j < hor_Sobel.n_cols; ++j) {
                auto x = hor_Sobel(i, j);
                auto y = ver_Sobel(i, j);
                grad_abs(i, j) = sqrt(x * x + y * y);
                grad_angle(i, j) = atan2(y, x);
            }
        }

        //compute LBP values for each pixel
        auto lbp_matrix = image_matrix.unary_map(LBP());

        // compute HOG and LBP histograms
        vector<float> image_features;
        vector<float> lbp_features;

        int side_blocks = sqrt(n_blocks);
        int ver_block_size = grad_abs.n_rows / side_blocks;
        int hor_block_size = grad_abs.n_cols / side_blocks;
        for (int i = 0; i < side_blocks; ++i) {
            for (int j = 0; j < side_blocks; ++j) {

                // define block borders
                int start_row = ver_block_size * i;
                int start_col = hor_block_size * j;
                int block_rows = (i == side_blocks - 1) ? grad_abs.n_rows - start_row : ver_block_size;
                int block_cols = (j == side_blocks - 1) ? grad_abs.n_cols - start_col : hor_block_size;

                // cut the block out
                auto block_abs = grad_abs.submatrix(start_row, start_col, block_rows, block_cols);
                auto block_angle = grad_angle.submatrix(start_row, start_col, block_rows, block_cols);
                auto block_lbp = lbp_matrix.submatrix(start_row, start_col, block_rows, block_cols);

                // create HOG histogram vector
                vector<float> hog_histogram(n_segments, 0.0);
                for (uint row = 0; row < block_angle.n_rows; ++row) {
                    for (uint col = 0; col < block_angle.n_cols; ++col) {
                        int segment = floor(block_angle(row, col) / (2 * M_PI / n_segments));
                        if (segment == n_segments / 2) {
                            segment = -segment;
                        }
                        segment += n_segments / 2;
                        hog_histogram[segment] += block_abs(row, col);
                    }
                }

                // compute the HOG histogram norm
                float hog_histogram_norm = 0.0;
                for (auto iter = hog_histogram.begin(); iter != hog_histogram.end(); ++iter) {
                    hog_histogram_norm += (*iter) * (*iter);
                }
                hog_histogram_norm = hog_histogram_norm > 0.000001 ? sqrt(hog_histogram_norm) : 1.0;

                // normalize the HOG histogram and add features
                for (auto iter = hog_histogram.begin(); iter != hog_histogram.end(); ++iter) {
                    *iter /= hog_histogram_norm;
                    image_features.push_back(*iter);
                }

                // compute LBP histogram vector
                vector<float> lbp_histogram(256, 0.0);
                for (uint row = 0; row < block_lbp.n_rows; ++row) {
                    for (uint col = 0; col < block_lbp.n_cols; ++col) {
                        ++lbp_histogram[block_lbp(row, col)];
                    }
                }

                // compute the LBP histogram norm
                float lbp_histogram_norm = 0.0;
                for (auto iter = lbp_histogram.begin(); iter != lbp_histogram.end(); ++iter) {
                    lbp_histogram_norm += (*iter) * (*iter);
                }
                lbp_histogram_norm = lbp_histogram_norm > 0.000001 ? sqrt(lbp_histogram_norm) : 1.0;

                // normalize the LBP histogram
                for (auto iter = lbp_histogram.begin(); iter != lbp_histogram.end(); ++iter) {
                    *iter /= lbp_histogram_norm;
                    lbp_features.push_back(*iter);
                }
            }
        }

        // concatenate LBP features with HOG features
        for (auto iter = lbp_features.begin(); iter != lbp_features.end(); ++iter) {
            image_features.push_back(*iter);
        }

        // color features
        const int color_side_blocks = 8;
        vector<float> color_features;
        uint color_ver_block_size = image->TellHeight() / color_side_blocks;
        uint color_hor_block_size = image->TellWidth() / color_side_blocks;


        for (int i = 0; i < color_side_blocks; ++i) {
            for (int j = 0; j < color_side_blocks; ++j) {
                // define block borders
                uint start_row = color_ver_block_size * i;
                uint start_col = color_hor_block_size * j;
                uint block_rows = (i == side_blocks - 1) ? image->TellHeight() - start_row : color_ver_block_size;
                uint block_cols = (j == side_blocks - 1) ? image->TellWidth() - start_col : color_hor_block_size;

                // calculate average color values
                float avg_red = 0.0, avg_green = 0.0, avg_blue = 0.0;
                for (uint row = start_row; row < start_row + color_ver_block_size; ++row) {
                    for (uint col = start_col; col < start_col + color_hor_block_size; ++col) {
                        auto pixel = image->GetPixel(col, row);
                        avg_red += pixel.Red;
                        avg_green += pixel.Green;
                        avg_blue += pixel.Blue;
                    }
                }
                int area = block_cols * block_rows;
                avg_red /= area * 255;
                avg_green /= area * 255;
                avg_blue /= area * 255;
                color_features.push_back(avg_red);
                color_features.push_back(avg_green);
                color_features.push_back(avg_blue);
            }
        }

        // concatenate color features with HOG features
        for (auto iter = color_features.begin(); iter != color_features.end(); ++iter) {
            image_features.push_back(*iter);
        }

        features->push_back(make_pair(image_features, data_set[image_idx].second));
    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here


    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");

        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}
