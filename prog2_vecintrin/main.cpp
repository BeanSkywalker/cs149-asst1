#include <stdio.h>
#include <algorithm>
#include <getopt.h>
#include <math.h>
#include "CS149intrin.h"
#include "logger.h"
using namespace std;

#define EXP_MAX 10

Logger CS149Logger;

void usage(const char* progname);
void initValue(float* values, int* exponents, float* output, float* gold, unsigned int N);
void absSerial(float* values, float* output, int N);
void absVector(float* values, float* output, int N);
void clampedExpSerial(float* values, int* exponents, float* output, int N);
void clampedExpVector(float* values, int* exponents, float* output, int N);
float arraySumSerial(float* values, int N);
float arraySumVector(float* values, int N);
bool verifyResult(float* values, int* exponents, float* output, float* gold, int N);

int main(int argc, char * argv[]) {
  int N = 16;
  bool printLog = false;

  // parse commandline options ////////////////////////////////////////////
  int opt;
  static struct option long_options[] = {
    {"size", 1, 0, 's'},
    {"log", 0, 0, 'l'},
    {"help", 0, 0, '?'},
    {0 ,0, 0, 0}
  };

  while ((opt = getopt_long(argc, argv, "s:l?", long_options, NULL)) != EOF) {

    switch (opt) {
      case 's':
        N = atoi(optarg);
        if (N <= 0) {
          printf("Error: Workload size is set to %d (<0).\n", N);
          return -1;
        }
        break;
      case 'l':
        printLog = true;
        break;
      case '?':
      default:
        usage(argv[0]);
        return 1;
    }
  }


  float* values = new float[N+VECTOR_WIDTH];
  int* exponents = new int[N+VECTOR_WIDTH];
  float* output = new float[N+VECTOR_WIDTH];
  float* gold = new float[N+VECTOR_WIDTH];
  initValue(values, exponents, output, gold, N);

  clampedExpSerial(values, exponents, gold, N);
  clampedExpVector(values, exponents, output, N);

  // absSerial(values, gold, N);
  // absVector(values, output, N);

  printf("\e[1;31mCLAMPED EXPONENT\e[0m (required) \n");
  bool clampedCorrect = verifyResult(values, exponents, output, gold, N);
  if (printLog) CS149Logger.printLog();
  CS149Logger.printStats();

  printf("************************ Result Verification *************************\n");
  if (!clampedCorrect) {
    printf("@@@ Failed!!!\n");
  } else {
    printf("Passed!!!\n");
  }

  printf("\n\e[1;31mARRAY SUM\e[0m (bonus) \n");
  if (N % VECTOR_WIDTH == 0) {
    float sumGold = arraySumSerial(values, N);
    float sumOutput = arraySumVector(values, N);
    float epsilon = 0.1;
    bool sumCorrect = abs(sumGold - sumOutput) < epsilon * 2;
    if (!sumCorrect) {
      printf("Expected %f, got %f\n.", sumGold, sumOutput);
      printf("@@@ Failed!!!\n");
    } else {
      printf("Passed!!!\n");
    }
  } else {
    printf("Must have N %% VECTOR_WIDTH == 0 for this problem (VECTOR_WIDTH is %d)\n", VECTOR_WIDTH);
  }

  delete [] values;
  delete [] exponents;
  delete [] output;
  delete [] gold;

  return 0;
}

void usage(const char* progname) {
  printf("Usage: %s [options]\n", progname);
  printf("Program Options:\n");
  printf("  -s  --size <N>     Use workload size N (Default = 16)\n");
  printf("  -l  --log          Print vector unit execution log\n");
  printf("  -?  --help         This message\n");
}

void initValue(float* values, int* exponents, float* output, float* gold, unsigned int N) {

  for (unsigned int i=0; i<N+VECTOR_WIDTH; i++)
  {
    // random input values
    values[i] = -1.f + 4.f * static_cast<float>(rand()) / RAND_MAX;
    exponents[i] = rand() % EXP_MAX;
    output[i] = 0.f;
    gold[i] = 0.f;
  }

}

bool verifyResult(float* values, int* exponents, float* output, float* gold, int N) {
  int incorrect = -1;
  float epsilon = 0.00001;
  for (int i=0; i<N+VECTOR_WIDTH; i++) {
    if ( abs(output[i] - gold[i]) > epsilon ) {
      incorrect = i;
      break;
    }
  }

  if (incorrect != -1) {
    if (incorrect >= N)
      printf("You have written to out of bound value!\n");
    printf("Wrong calculation at value[%d]!\n", incorrect);
    printf("value  = ");
    for (int i=0; i<N; i++) {
      printf("% f ", values[i]);
    } printf("\n");

    printf("exp    = ");
    for (int i=0; i<N; i++) {
      printf("% 9d ", exponents[i]);
    } printf("\n");

    printf("output = ");
    for (int i=0; i<N; i++) {
      printf("% f ", output[i]);
    } printf("\n");

    printf("gold   = ");
    for (int i=0; i<N; i++) {
      printf("% f ", gold[i]);
    } printf("\n");
    return false;
  }
  printf("Results matched with answer!\n");
  return true;
}

// computes the absolute value of all elements in the input array
// values, stores result in output
void absSerial(float* values, float* output, int N) {
  for (int i=0; i<N; i++) {
    float x = values[i];
    if (x < 0) {
      output[i] = -x;
    } else {
      output[i] = x;
    }
  }
}


// implementation of absSerial() above, but it is vectorized using CS149 intrinsics
void absVector(float* values, float* output, int N) {
  __cs149_vec_float x;
  __cs149_vec_float result;
  __cs149_vec_float zero = _cs149_vset_float(0.f);
  __cs149_mask maskAll, maskIsNegative, maskIsNotNegative;

//  Note: Take a careful look at this loop indexing.  This example
//  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
//  Why is that the case?
  for (int i=0; i<N; i+=VECTOR_WIDTH) {

    // All ones
    maskAll = _cs149_init_ones();

    // All zeros
    maskIsNegative = _cs149_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _cs149_vload_float(x, values+i, maskAll);               // x = values[i];

    // Set mask according to predicate
    _cs149_vlt_float(maskIsNegative, x, zero, maskAll);     // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _cs149_vsub_float(result, zero, x, maskIsNegative);      //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _cs149_mask_not(maskIsNegative);     // } else {

    // Execute instruction ("else" clause)
    _cs149_vload_float(result, values+i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _cs149_vstore_float(output+i, result, maskAll);
  }
}


// accepts an array of values and an array of exponents
//
// For each element, compute values[i]^exponents[i] and clamp value to
// 9.999.  Store result in output.
void clampedExpSerial(float* values, int* exponents, float* output, int N) {
  for (int i=0; i<N; i++) {
    float x = values[i];
    int y = exponents[i];
    if (y == 0) {
      output[i] = 1.f;
    } else {
      float result = x;
      int count = y - 1;
      while (count > 0) {
        result *= x;
        count--;
      }
      if (result > 9.999999f) {
        result = 9.999999f;
      }
      output[i] = result;
    }
  }
}

void print_vec_int (__cs149_vec_int & value) {
  printf("printf int : ");
  for (size_t i = 0; i < VECTOR_WIDTH; i++)
  {
    printf("%9d ", value.value[i]);
  }
  printf("\n");
}

void print_vec_float (__cs149_vec_float& value) {
  printf("printf float : ");
  for (size_t i = 0; i < VECTOR_WIDTH; i++)
  {
    printf("% f ", value.value[i]);
  }
  printf("\n");
}

void print_vec_mask (__cs149_mask& value) {
  printf("printf mask : ");
  for (size_t i = 0; i < VECTOR_WIDTH; i++)
  {
    if(value.value[i]) {
      printf("         1");
    } else {
      printf("         0");
    }
    
  }
  printf("\n");
}

void print_data_float (float* value) {
  printf("printf data float : ");
  for (size_t i = 0; i < VECTOR_WIDTH; i++)
  {
    printf("% f ", value[i]);
  }
  printf("\n");
}

void clampedExpVector(float* values, int* exponents, float* output, int N) {

  //
  // CS149 STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

  int remain_n  = N;
  int value_ptr = 0;
  int remain_check;

  __cs149_mask        vec_mask;
  __cs149_mask        vec_mask_data_cpy;
  __cs149_mask        vec_mask_all1;
  __cs149_mask        vec_check_mask;
  __cs149_vec_float   vec_exp_result;
  __cs149_vec_float   vec_max_value;
  __cs149_vec_float   vec_exp_value;
  __cs149_vec_int     vec_exponents;
  __cs149_vec_int     vec_all_0;
  __cs149_vec_int     vec_all_1;

  float               value_1   = 1;
  float               max_value = 9.999999f;

  

  vec_all_0 = _cs149_vset_int( 0 );
  vec_all_1 = _cs149_vset_int( 1 );

  vec_max_value = _cs149_vset_float(max_value);

  vec_mask_all1 = _cs149_init_ones(VECTOR_WIDTH);


  while (remain_n > 0) {
    // setting mask values
      if (remain_n < VECTOR_WIDTH ) {
        vec_mask          = _cs149_init_ones(remain_n);
        vec_mask_data_cpy = _cs149_init_ones(remain_n);
      } else {
        vec_mask          = _cs149_init_ones(VECTOR_WIDTH);
        vec_mask_data_cpy = _cs149_init_ones(VECTOR_WIDTH);
      }
      
    // loading values
      _cs149_vload_float  (vec_exp_value  , &values[N - remain_n]    , vec_mask);
      _cs149_vload_float  (vec_exp_result , &values[N - remain_n]    , vec_mask);
      // _cs149_vset_float   (vec_exp_result , value_1                 , vec_mask);
      _cs149_vload_int    (vec_exponents  , &exponents[N - remain_n] , vec_mask);
      // printf("->load value\r\n");
      // print_vec_float(vec_exp_value);
      // print_vec_int(vec_exponents);
      // print_vec_mask(vec_mask);

    // start calculation
      // check 0 exp
        // printf("->check 0 exp\r\n");
        // print_vec_mask(vec_mask);
        
        _cs149_veq_int    ( vec_check_mask , vec_exponents , vec_all_0 , vec_mask);
        _cs149_vset_float ( vec_exp_result , value_1 , vec_check_mask);

        // print_vec_mask(vec_check_mask);
        vec_check_mask = _cs149_mask_not ( vec_check_mask );
        // print_vec_mask(vec_check_mask);
        vec_mask = _cs149_mask_and( vec_mask , vec_check_mask);
        // print_vec_mask(vec_mask);

      // exp value -1, which means execute '*' start from 2
        _cs149_vsub_int   ( vec_exponents , vec_exponents , vec_all_1 , vec_mask);
        // update mask data
          _cs149_veq_int    ( vec_check_mask , vec_exponents , vec_all_0 , vec_mask);
          vec_check_mask = _cs149_mask_not ( vec_check_mask );
          vec_mask = _cs149_mask_and( vec_mask , vec_check_mask);
      
        // print_vec_float(vec_exp_result);
        // print_vec_int(vec_exponents);
        // print_vec_mask(vec_mask);
      // while loop
        while ( 0 < _cs149_cntbits(vec_mask)) {
          
            // implement multiply
            _cs149_vmult_float  (vec_exp_result , vec_exp_result , vec_exp_value , vec_mask);

          // update the exp value
            _cs149_vsub_int   ( vec_exponents , vec_exponents , vec_all_1 , vec_mask);
            _cs149_veq_int    ( vec_check_mask , vec_exponents , vec_all_0 , vec_mask);
            vec_check_mask = _cs149_mask_not ( vec_check_mask );
            vec_mask = _cs149_mask_and( vec_mask , vec_check_mask);

          // print_vec_float(vec_exp_result);
          // print_vec_int(vec_exponents);
          // print_vec_mask(vec_mask);
        }
      // ckeck if > 9.99
        // printf("->check max value \r\n");
        // print_vec_float(vec_exp_result);
        _cs149_vgt_float (vec_check_mask , vec_exp_result , vec_max_value , vec_mask_all1);
        // print_vec_mask(vec_check_mask);
        _cs149_vset_float(vec_exp_result , max_value, vec_check_mask);
        // print_vec_float(vec_exp_result);
      // copy output value
        _cs149_vstore_float( &output[ N - remain_n ], vec_exp_result , vec_mask_data_cpy);
      
        remain_n = remain_n - VECTOR_WIDTH;

        // printf("-> remain N number: %d", remain_n);
  }
  
}

// returns the sum of all elements in values
float arraySumSerial(float* values, int N) {
  float sum = 0;
  for (int i=0; i<N; i++) {
    sum += values[i];
  }

  return sum;
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float* values, int N) {
  
  //
  // CS149 STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  int remain_n  = N;
  int value_ptr = 0;
  int remain_check;


  __cs149_mask        vec_mask_all1;
  __cs149_mask        vec_mask_first_1;
  __cs149_vec_float   vec_sum_result;
  __cs149_vec_float   vec_sum_value;

  __cs149_vec_float   vec_max_value;
  __cs149_vec_int     vec_all_0;
  __cs149_vec_int     vec_all_1;
  

  float               value_1   = 1;
  float               max_value = 9.999999f;


  vec_all_0 = _cs149_vset_int( 0 );
  vec_all_1 = _cs149_vset_int( 1 );

  vec_max_value = _cs149_vset_float(max_value);

  vec_mask_all1 = _cs149_init_ones(VECTOR_WIDTH);

  vec_mask_first_1 = _cs149_init_ones (1);


  // initial value
    _cs149_vset_float ( vec_sum_result , .0f , vec_mask_all1);

  // add value by vectors
    for (int i=0; i<N; i+=VECTOR_WIDTH) {
      _cs149_vload_float(vec_sum_value  , &values[i]     , vec_mask_all1);
      _cs149_vadd_float (vec_sum_result , vec_sum_result , vec_sum_value , vec_mask_all1);
      // print_data_float  (&values[i]);
      // print_vec_float   (vec_sum_result);
    }

  // summation value insize a vector
    int i=VECTOR_WIDTH>>1;

    // printf("-> adding inside a vector\r\n");
    while (i > 0)
    {
      _cs149_hadd_float(vec_sum_result , vec_sum_result);
      _cs149_interleave_float(vec_sum_result , vec_sum_result);
      i = i/2;
      // print_vec_float   (vec_sum_result);
    }

    float out_temp;

    _cs149_vstore_float( &out_temp , vec_sum_result , vec_mask_first_1);
    

  return out_temp;
}

