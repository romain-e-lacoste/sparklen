// Author : Romain E. Lacoste
// License : BSD-3-Clause

#ifndef LIB_INCLUDE_SPARKLEN_ARRAY_ARRAY_H_
#define LIB_INCLUDE_SPARKLEN_ARRAY_ARRAY_H_

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>


template <typename T> class SharedArray; // Forward declaration

template <typename T> class Array {

	private:

    size_t _size;
    T *_data;
    bool _ownData;

	public:

    // Default constructor
    Array();

    // Constructor with size and optional external data
    Array(size_t size, T *data = nullptr);

    // Copy constructor
    Array(const Array<T> &other);

    // Move constructor
    Array(Array<T> &&other) noexcept;

    // Destructor
    virtual ~Array();

    // Copy assignment operator
    Array<T> &operator=(const Array<T> &other);

    // Move assignment operator
    Array<T> &operator=(Array<T> &&other) noexcept;

    // Access elements operator
    T &operator[](const size_t i) const;

    // Utility methods
    size_t size() const;
    T *data() const;
    void _print() const;
    bool own() const;

    // Arithmetic operations
    void operator+=(T scalar);
    void operator*=(T scalar);
    void operator/=(T scalar);

    // Array addition
    void add(const Array<T> &arr);

    // Friend class declaration for SharedArray
    friend class SharedArray<T>; // Declare SharedArray as a friend

};

// Default constructor
template <typename T> Array<T>::Array() {
    _size = 0;
    _data = nullptr;
    _ownData = false;
}

// Constructor with size and optional external data
template <typename T> Array<T>::Array(size_t size, T *data) {
    _size = size;
    if (data == nullptr) {
        _ownData = true;
        _data = new T[_size];
        for (size_t i = 0; i < _size; i++) {
            _data[i] = 0;
        }
    } else {
        _ownData = false;
        _data = data;
    }
}

// Copy constructor
template <typename T> Array<T>::Array(const Array<T> &other) {
    _size = other._size;
    _data = new T[_size];
    _ownData = true;
    std::copy(other._data, other._data + _size, _data);
}

// Move constructor
template <typename T> Array<T>::Array(Array<T> &&other) noexcept {
    _size = other._size;
    _data = other._data;
    _ownData = other._ownData;
    other._size = 0;
    other._data = nullptr;
    other._ownData = false;
}

// Destructor
template <typename T> Array<T>::~Array() {
    if (_ownData) {
        delete[] _data;
    }
}

// Copy assignment operator
template <typename T> Array<T> &Array<T>::operator=(const Array<T> &other) {
    if (this != &other) {
        // Clean up existing resources
        if (_ownData) {
            delete[] _data;
        }
        // Allocate new resources and copy data
        _size = other._size;
        _data = new T[_size];
        _ownData = true;
        std::copy(other._data, other._data + _size, _data);
    }
    return *this;
}

// Move assignment operator
template <typename T> Array<T> &Array<T>::operator=(Array<T> &&other) noexcept {
    if (this != &other) {
        // Clean up existing resources
        if (_ownData) {
            delete[] _data;
        }
        // Steal resources from other
        _size = other._size;
        _data = other._data;
        _ownData = other._ownData;
        // Reset other
        other._size = 0;
        other._data = nullptr;
        other._ownData = false;
    }
    return *this;
}

// Access elements
template <typename T> T &Array<T>::operator[](const size_t i) const {
    if (i >= _size) throw std::out_of_range("Index out of range");
    return _data[i];
}

// Get size of the array
template <typename T> size_t Array<T>::size() const {
    return _size;
}

// Get pointer to data
template <typename T> T *Array<T>::data() const {
    return _data;
}

// Print array elements
template <typename T> void Array<T>::_print() const {
    for (size_t i = 0; i < _size; ++i) {
        std::cout << " " << _data[i];
    }
    std::cout << std::endl;
}

template <typename T> bool Array<T>::own() const {
    return _ownData;
}

// Add scalar to each element
template <typename T> void Array<T>::operator+=(T scalar) {
    for (size_t i = 0; i < _size; ++i) {
        _data[i] += scalar;
    }
}

// Multiply each element by scalar
template <typename T> void Array<T>::operator*=(T scalar) {
    for (size_t i = 0; i < _size; ++i) {
        _data[i] *= scalar;
    }
}

// Divide each element by scalar
template <typename T> void Array<T>::operator/=(T scalar) {
    if (scalar == T()) throw std::invalid_argument("Division by zero.");
    for (size_t i = 0; i < _size; ++i) {
        _data[i] /= scalar;
    }
}

template <typename T> void Array<T>::add(const Array<T> &arr) {
    if (_size != arr.size()) throw std::invalid_argument("Arrays must have the same size.");
    for (size_t i = 0; i < _size; ++i) {
        _data[i] += arr._data[i];
    }
}

#define ARRAY_DEFINE_TYPE(TYPE, NAME)                 \
  typedef Array<TYPE> Array##NAME##1D;                    \
  typedef std::vector<Array##NAME##1D> ListArray##NAME##1D; \
  typedef std::vector<ListArray##NAME##1D> ListListArray##NAME##1D

ARRAY_DEFINE_TYPE(double, Double);
ARRAY_DEFINE_TYPE(int, Int);

#undef ARRAY_DEFINE_TYPE


#endif /* LIB_INCLUDE_SPARKLEN_ARRAY_ARRAY_H_ */
