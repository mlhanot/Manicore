#ifndef PARALLEL_FOR
#define PARALLEL_FOR

#include <thread>

/** \file parallel_for.hpp
   Helper to parallelize operations using pthreads.
*/
namespace Manicore
{

  /// Function to distribute elements (considered as jobs) over threads. It returns a pair of vectors indicating the start and end element of each thread
  static std::pair<std::vector<int>, std::vector<int>> 
    distributeLoad(size_t nb_elements, unsigned nb_threads)
  { 
    // Vectors of start and end indices
    std::vector<int> start(nb_threads);
    std::vector<int> end(nb_threads);

    // Compute the batch size and the remainder
    unsigned batch_size = nb_elements / nb_threads;
    unsigned batch_remainder = nb_elements % nb_threads;

    // Distribute the remainder over the threads to get the start and end indices for each thread
    for (unsigned i = 0; i < nb_threads; ++i) {
      if (i < batch_remainder){
        start[i] = i * batch_size + i;
        end[i] = start[i] + batch_size + 1;
      }
      else{
        start[i] = i * batch_size + batch_remainder;
        end[i] = start[i] + batch_size;
      }
    }

    return std::make_pair(start, end);
  }

  /// Generic function to execute threaded processes
  static inline void parallel_for(unsigned nb_elements,
                                  std::function<void(size_t start, size_t end)> functor,
                                  bool use_threads = true)
  {
    unsigned nb_threads_hint = std::thread::hardware_concurrency();
    unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

    // Generate the start and end indices
    auto [start, end] = distributeLoad(nb_elements, nb_threads);

    std::vector<std::thread> my_threads(nb_threads);

    if (use_threads) {
      // Multithread execution
      for (unsigned i = 0; i < nb_threads; ++i) {
          my_threads[i] = std::thread(functor, start[i], end[i]);
      }
    } else {
      // Single thread execution (for easy debugging)
      for(unsigned i = 0; i < nb_threads; ++i) {
          functor(start[i], end[i]);
      }
    }

    // Wait for the other thread to finish their task
    if (use_threads) {
      std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    }
  }

  
} // end of namespace
#endif
