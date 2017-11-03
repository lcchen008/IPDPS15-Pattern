#ifndef LIB_UTIL_H_
#define LIB_UTIL_H_

#include "common.h"
#include <string>
#include <sstream>
#include <iostream>
#include <math.h>

class StringJoin
{
  const std::string sep;
  bool first;
  std::ostringstream ss;
 public:
  StringJoin(const std::string &sep=", "): sep(sep), first(true) {}
  void append(const std::string &s) {
    if (!first) ss << sep;
    ss << s;
    first = false;
  }
        
  template <class T>
  std::ostringstream &operator<< (const T &s) {
    if (!first) ss << sep;
    ss << s;
    first = false;
    return ss;
  }
  std::string get() const {
    return ss.str();
  }
  std::string str() const {
    return get();
  }
};

inline std::ostream &operator<<(std::ostream &os, const StringJoin &sj)
{
	os<<sj.get();
	return os;
}


#define __FUNC_ID__ __FUNCTION__

#define LOGGING_FILE_BASENAME(name)                                     \
  ((std::string(name).find_last_of('/') == std::string::npos) ?         \
   std::string(name) : std::string(name).substr(std::string(name).find_last_of('/')+1))

#if defined(PS_DEBUG)
#define LOG_DEBUG()                             \
	  (std::cerr << "[DEBUG:" << __FUNC_ID__        \
	      << "@" << LOGGING_FILE_BASENAME(__FILE__)    \
	      << "#"  << __LINE__ << "] ")
#else
#define LOG_DEBUG()  if (0) std::cerr
#endif


#endif

