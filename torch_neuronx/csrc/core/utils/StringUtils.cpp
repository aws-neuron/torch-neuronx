#include "StringUtils.h"

namespace at::neuron::utils {

std::string SimplifyForFilename(const std::string& str) {
  std::string result;
  bool prev_underscore = false;

  for (char c : str) {
    if (c == '<' || c == '>' || c == ':' || c == '"' || c == '/' || c == '\\' || c == '|' ||
        c == '?' || c == '*' || c == ' ' || c == '\'' || c == '(' || c == ')' || c == '[' ||
        c == ']' || c == ',') {
      if (!prev_underscore) {
        result += '_';
        prev_underscore = true;
      }
    } else {
      result += c;
      prev_underscore = false;
    }
  }

  // Remove leading/trailing underscores
  while (!result.empty() && result.front() == '_') result.erase(0, 1);
  while (!result.empty() && result.back() == '_') result.pop_back();

  return result;
}

}  // namespace at::neuron::utils
