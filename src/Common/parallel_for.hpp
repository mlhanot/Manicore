#ifndef PARALLEL_FOR
#define PARALLEL_FOR

#include <thread>
#include <forward_list>
#include <Eigen/Sparse>

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

  /// Function to assemble a global sparse matrix from a procedure that compute local contributions
  template<typename FType>
  Eigen::SparseMatrix<double> parallel_assembly(
      size_t nb_elements /*!< Number of elements over which the threading will be done */,
      std::pair<size_t,size_t> systemSize /*!< Number of rows and columns of the matrix to assemble */,
      FType localAssembly /*!< Functor performing the local assembly in a single cell */,
      bool use_threads = true /*!< Determine if threads must be used or not */)
  {
    std::forward_list<Eigen::Triplet<double>> triplets;
    if (use_threads) {
      // Select the number of threads
      unsigned nb_threads_hint = std::thread::hardware_concurrency();
      unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);
      // Generate the start and end indices
      auto [start, end] = distributeLoad(nb_elements, nb_threads);
      // Create vectors of triplets 
      std::vector<std::forward_list<Eigen::Triplet<double>>> tripletsVect(nb_threads);

      // Assign a task to each thread
      std::vector<std::thread> my_threads(nb_threads);
      for (unsigned i = 0; i < nb_threads; ++i) {
          my_threads[i] = std::thread(localAssembly, start[i], end[i], &tripletsVect[i]);
      }
      // Wait for the other threads to finish their task
      std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
      // Join triplets
      for (unsigned i = 0; i < nb_threads; ++i) {
        triplets.splice_after(triplets.cbefore_begin(),tripletsVect[i]);
      }
    } else { // Single thread
      localAssembly(0,nb_elements,&triplets);
    }
    // Construct the system
    Eigen::SparseMatrix<double> system(systemSize.first,systemSize.second);
    system.setFromTriplets(triplets.cbegin(),triplets.cend());
    return system;
  }
  
} // end of namespace
#endif
