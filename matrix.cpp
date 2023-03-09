#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <thread>
#include <omp.h>
#include <pthread.h>
#include <unistd.h>


//定义矩阵数据结构
struct Matrix {
    int rows;
    int cols;
    std::vector<int> data;

    //构造函数，初始化数据
    Matrix(int r, int c) : rows(r), cols(c), data(r * c) {}

    //获取矩阵元素
    int& operator()(int row, int col) {
        return data[row * cols + col];
    }

    //获取矩阵元素
    int operator()(int row, int col) const {
        return data[row * cols + col];
    }

    //打印矩阵
    void print() const {
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                std::cout << operator()(i,j) << " ";
            }
            std::cout << std::endl;
        }
    }
};

//随机生成矩阵
Matrix generate_matrix(int rows, int cols) {
    Matrix m(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            m(i,j) = dis(gen);
        }
    }
    return m;
}

//sequential
void multiply_sequential(const Matrix& A, const Matrix& B, Matrix& C) {
    auto start_time = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < A.rows; ++i) {
        for(int j = 0; j < B.cols; ++j) {
            int sum = 0;
            for(int k = 0; k < A.cols; ++k) {
                sum += A(i,k) * B(k,j);
            }
            C(i,j) = sum;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "multiply_thread time: " << time << " microseconds" << std::endl;
}

//C++ standard library thread
void multiply_thread(const Matrix& A, const Matrix& B, Matrix& C) {
    std::vector<std::thread> threads;
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = A.rows / num_threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    //assign tasks
    for(int t = 0; t < num_threads; ++t) {
        threSads.emplace_back([&A, &B, &C](int begin, int end) {
            for(int i = begin; i < end; ++i) {
                for(int j = 0; j < B.cols; ++j) {
                    int sum = 0;
                    for(int k = 0; k < A.cols; ++k) {
                        sum += A(i,k) * B(k,j);
                    }
                    C(i,j) = sum;
                }
            }
        }, t * chunk_size, (t + 1) * chunk_size);
    }

    //wait threads
    for(auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "multiply_thread time: " << time << " microseconds" << std::endl;
}

//OpenMP
void multiply_omp(const Matrix& A, const Matrix& B, Matrix& C) {
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for(int i = 0; i < A.rows; ++i) {
        for(int j = 0; j < B.cols; ++j) {
            int sum = 0;
            for(int k = 0; k < A.cols; ++k) {
                sum += A(i,k) * B(k,j);
            }
            C(i,j) = sum;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "multiply_omp time: " << time << " microseconds" << std::endl;
}

//subfunction of pthread
void* multiply_thread_func(void* args) {
    auto& params = *(std::tuple<int, int, int, const Matrix*, const Matrix*, Matrix*>*)args;
    int begin = std::get<0>(params);
    int end = std::get<1>(params);
    int cols = std::get<2>(params);
    const Matrix& A = *std::get<3>(params);
    const Matrix& B = *std::get<4>(params);
    Matrix& C = *std::get<5>(params);

    for(int i = begin; i < end; ++i) {
        for(int j = 0; j < cols; ++j) {
            int sum = 0;
            for(int k = 0; k < A.cols; ++k) {
                sum += A(i,k) * B(k,j);
            }
            C(i,j) = sum;
        }
    }

    pthread_exit(nullptr);
}

//pthread
void multiply_pthread(const Matrix& A, const Matrix& B, Matrix& C) {
    std::vector<pthread_t> threads;
    int num_threads = sysconf(_SC_NPROCESSORS_ONLN);// num_threads = 8
    int chunk_size = A.rows / num_threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    //assign task
    std::vector<std::tuple<int, int, int, const Matrix*, const Matrix*, Matrix*>> params(num_threads);
    int begin = 0;
    for(int t = 0; t < num_threads; ++t) {
        int end = (t == num_threads - 1) ? A.rows : (begin + chunk_size);
        std::get<0>(params[t]) = begin;
        std::get<1>(params[t]) = end;
        std::get<2>(params[t]) = B.cols;
        std::get<3>(params[t]) = &A;
        std::get<4>(params[t]) = &B;
        std::get<5>(params[t]) = &C;
        pthread_t thread;
        pthread_create(&thread, nullptr, multiply_thread_func, &params[t]);
        threads.push_back(thread);
        begin = end;
    }

    //wait threads
    for(auto& thread : threads) {
        pthread_join(thread, nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "multiply_pthread time: " << time << " microseconds" << std::endl;
}

int main() {
    int size = 1000;

    //随机生成矩阵
    Matrix A = generate_matrix(size, size);
    Matrix B = generate_matrix(size, size);
    Matrix C(size, size);

    //计算矩阵乘法并输出时间
    multiply_sequential(A,B,C);
    multiply_thread(A, B, C);
    multiply_omp(A, B, C);
    multiply_pthread(A, B, C);

    return 0;
}
