#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  // **Solution: Change Mask**

  __pp_vec_float reg_val, result, exp_result;
  __pp_vec_int reg_exp, zero_int, ones_int, count_int;
  __pp_vec_float upperBound_float;

  __pp_mask maskAll, maskIsZero, maskIsNotZero, maskCountIsNotZero, maskBeyondBound;
  float upperBound = 9.999999f;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    int next_pos = i + VECTOR_WIDTH;
    if (next_pos > N) {
      maskAll = _pp_init_ones(N - i);
    } else {
      maskAll = _pp_init_ones();
    }
    // maskAll = _pp_init_ones();

    maskIsZero = _pp_init_ones(0);
    maskCountIsNotZero = _pp_init_ones(0);
    maskBeyondBound = _pp_init_ones(0);
    zero_int = _pp_vset_int(0);
    ones_int = _pp_vset_int(1);
    count_int = _pp_vset_int(0);
    upperBound_float = _pp_vset_float(upperBound);

    _pp_vload_float(reg_val, values + i, maskAll); // reg_val = { values[i], values[i + V_width] };
    _pp_vload_int(reg_exp, exponents + i, maskAll); // reg_exp = { exponents[i], exponents[i + V_width] };
    
    _pp_veq_int(maskIsZero, reg_exp, zero_int, maskAll); // if (y == 0) {
    _pp_vset_float(result, 1.f, maskIsZero); // output[i] = 1.f }

    maskIsNotZero = _pp_mask_not(maskIsZero); // else {
    _pp_vload_float(result, values + i, maskIsNotZero);// float result = x;
    _pp_vsub_int(count_int, reg_exp, ones_int, maskIsNotZero);// int count = y - 1;
    
    while (1) { // while (count > 0) {
      _pp_vgt_int(maskCountIsNotZero, count_int, zero_int, maskIsNotZero);
      int NotZero = _pp_cntbits(maskCountIsNotZero);
      if (NotZero == 0) {
        break;
      }

      _pp_vmult_float(result, result, reg_val, maskCountIsNotZero); // result *= x
      _pp_vsub_int(count_int, count_int, ones_int, maskCountIsNotZero);// count-- }
    }
    _pp_vgt_float(maskBeyondBound, result, upperBound_float, maskIsNotZero);// if (result > 9.9999999f) {
    _pp_vset_float(result, upperBound, maskBeyondBound); // result = 9.99999f }
    
    _pp_vstore_float(output + i, result, maskAll); // output[i] = result

  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
// Goal: O(N / VECTOR_WIDTH + log2(VECTOR_WIDTH))
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_vec_float sum, reg_val, tmp;
  __pp_mask maskAll;

  sum = _pp_vset_float(0.0);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    reg_val = _pp_vset_float(0.0);
    maskAll = _pp_init_ones();

    _pp_vload_float(reg_val, values + i, maskAll); // reg_val = { values[i], values[i + V_width] };
    _pp_vadd_float(sum, sum, reg_val, maskAll);
  }
  for (int i=1; i < VECTOR_WIDTH; i<<=1) {
    _pp_hadd_float(sum, sum);
    _pp_interleave_float(tmp, sum);
    _pp_vmove_float(sum, tmp, maskAll);  
  }
  // float tmp = new float(4);
  _pp_vstore_float(values, sum, maskAll);
  return values[0];
}