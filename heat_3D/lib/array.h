#ifndef LIB_ARRAY_H_
#define LIB_ARRAY_H_

#include <assert.h>
#include <numeric>
#include <functional>
#include "common.h"
#include "macro.h"
#include "util.h"
#include <vector>

#include <boost/array.hpp>

template <typename ty>
class IntegerArray: public boost::array<ty, MAX_DIM> {
  // TODO: necessary?
  //typedef boost::array<c, MAX_DIM> pt;
  //typedef ty elmt;
 public:  
  IntegerArray() {
    this->assign(0);
  }
  explicit IntegerArray(ty x) {
    //this->assign(0);
    //(*this)[0] = x;
    this->assign(x);
  }
  explicit IntegerArray(ty x, ty y) {
    this->assign(0);
    (*this)[0] = x;
    (*this)[1] = y;
  }
  explicit IntegerArray(ty x, ty y, ty z) {
    this->assign(0);    
    (*this)[0] = x;
    (*this)[1] = y;
    (*this)[2] = z;
  }

  template <typename ty2>
  IntegerArray(const ty2 *v) {
    for (int i = 0; i < MAX_DIM; ++i) {
      (*this)[i] = (ty)v[i];
    }
  }
  template <typename ty2>
  IntegerArray(const IntegerArray<ty2> &v) {
    for (int i = 0; i < MAX_DIM; ++i) {
      (*this)[i] = (ty)v[i];
    }
  }
  void Set(ty v) {
    this->assign(v);
  }

  //used to get size
  size_t accumulate(int len) const {
    size_t v = 1;
    FOREACH (it, this->begin(), this->begin() + len) {
      v *= *it;
    }
    return v;
  }
  std::ostream& print(std::ostream &os) const {
    StringJoin sj;
    FOREACH(i, this->begin(), this->end()) {
      sj << *i;
    }
    os << "{" << sj.str() << "}";
    return os;
  }
  IntegerArray<ty> operator+(const IntegerArray<ty> &x) const {
    IntegerArray<ty> ret;
    for (int i = 0; i < MAX_DIM; ++i) {
      ret[i] = (*this)[i] + x[i];
    }
    return ret;
  }
  IntegerArray<ty> operator+(const ty &x) const {
    IntegerArray ret;
    for (int i = 0; i < MAX_DIM; ++i) {
      ret[i] = (*this)[i] + x;
    }
    return ret;
  }
  IntegerArray<ty> operator-(const IntegerArray<ty> &x) const {
    IntegerArray<ty> ret;
    for (int i = 0; i < MAX_DIM; ++i) {
      ret[i] = (*this)[i] - x[i];
    }
    return ret;
  }    
  IntegerArray<ty> operator-(const ty &x) const {
    IntegerArray ret;
    for (int i = 0; i < MAX_DIM; ++i) {
      ret[i] = (*this)[i] - x;
    }
    return ret;
  }
  IntegerArray<ty> operator*(const ty &x) const {
    IntegerArray ret;
    for (int i = 0; i < MAX_DIM; ++i) {
      ret[i] = (*this)[i] * x;
    }
    return ret;
  }
  IntegerArray<ty> operator/(const ty &x) const {
    IntegerArray ret;
    for (int i = 0; i < MAX_DIM; ++i) {
      ret[i] = (*this)[i] / x;
    }
    return ret;
  }
  bool operator>(const ty &x) const {
    for (int i = 0; i < MAX_DIM; ++i) {
      if (!((*this)[i] > x)) return false;
    }
    return true;
  }
  bool operator<(const ty &x) const {
    for (int i = 0; i < MAX_DIM; ++i) {
      if (!((*this)[i] < x)) return false;
    }
    return true;
  }
  bool operator>=(const ty &x) const {
    for (int i = 0; i < MAX_DIM; ++i) {
      if (!((*this)[i] >= x)) return false;
    }
    return true;
  }
  bool operator<=(const ty &x) const {
    for (int i = 0; i < MAX_DIM; ++i) {
      if (!((*this)[i] <= x)) return false;
    }
    return true;
  }
  bool operator==(const ty &x) const {
    for (int i = 0; i < MAX_DIM; ++i) {
      if (!((*this)[i] == x)) return false;
    }
    return true;
  }
  bool operator!=(const ty &x) const {
    return !(*this == x);
  }
  IntegerArray<ty> &operator+=(const IntegerArray<ty> &x) {
    for (int i = 0; i < MAX_DIM; ++i) {
      (*this)[i] += x[i];
    }
    return *this;
  }
  IntegerArray<ty> &operator-=(const IntegerArray<ty> &x) {
    for (int i = 0; i < MAX_DIM; ++i) {
      (*this)[i] -= x[i];
    }
    return *this;
  }
  IntegerArray<ty> &operator*=(const IntegerArray<ty> &x) {
    for (int i = 0; i < MAX_DIM; ++i) {
      (*this)[i] *= x[i];
    }
    return *this;
  }
  IntegerArray<ty> &operator/=(const IntegerArray<ty> &x) {
    for (int i = 0; i < MAX_DIM; ++i) {
      (*this)[i] /= x[i];
    }
    return *this;
  }
  void SetNoLessThan(const ty &x) {
    for (int i = 0; i < MAX_DIM; ++i) {
      (*this)[i] = std::max((*this)[i], x);
    }
  }
  void SetNoLessThan(const IntegerArray<ty> &x) {
    for (int i = 0; i < MAX_DIM; ++i) {
      (*this)[i] = std::max((*this)[i], x[i]);
    }
  }
  void SetNoMoreThan(const ty &x) {
    for (int i = 0; i < MAX_DIM; ++i) {
      (*this)[i] = std::min((*this)[i], x);
    }
  }
  void SetNoMoreThan(const IntegerArray<ty> &x) {
    for (int i = 0; i < MAX_DIM; ++i) {
      (*this)[i] = std::min((*this)[i], x[i]);
    }
  }
  void Set(PSIndex *buf) const {
    for (int i = 0; i < MAX_DIM; ++i) {
      buf[i] = (*this)[i];
    }
  }
  bool LessThan(const IntegerArray<ty> &x, int num_dims) {
    for (int i = 0; i < num_dims; ++i) {
      if (!((*this)[i] < x[i])) return false;
    }
    return true;
  }
  bool GreaterThan(const IntegerArray<ty> &x, int num_dims) {
    for (int i = 0; i < num_dims; ++i) {
      if (!((*this)[i] > x[i])) return false;
    }
    return true;
  }
  template <typename ty2>
  IntegerArray<ty> operator=(const IntegerArray<ty2> x) {
    for (int i = 0; i < MAX_DIM; ++i) {
      (*this)[i] = (ty)x[i];
    }
    return *this;
  }
};

// REFACTORING: rename to IndexVector
typedef std::vector<int> IntVector;
typedef std::vector<size_t> SizeVector;

typedef IntegerArray<int> IntArray;
typedef IntegerArray<unsigned> UnsignedArray;
typedef IntegerArray<size_t> SizeArray;
typedef IntegerArray<ssize_t> SSizeArray;
typedef IntegerArray<int> IndexArray;

template <typename ty>
inline std::ostream &operator<<(
    std::ostream &os,
    const IntegerArray<ty> &x) {
  return x.print(os);
}

template <typename ty>
inline std::ostream &operator<<(std::ostream &os,
                                const std::vector<ty> &x) {
  StringJoin sj;
  FOREACH (i, x.begin(), x.end()) { sj << *i; }
  os << "{" << sj << "}";  
  return os;
}
#endif /* PHYSIS_INTERNAL_COMMON_H_ */
