#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
  }
  bool converged = false;
  double score_new[numNodes + 1];
  memset(score_new, 0, sizeof(score_new));

  while ( !converged ) {
    // iterate for all nodes
    for (int i = 0; i < numNodes; i++) {
      const Vertex* start = incoming_begin(g, i);
      const Vertex* end = incoming_end(g, i);
      
      double nsum = 0;
      // iterate for all incoming edges of node i
      for (const Vertex* v = start; v!=end; v++) {
        // sum up old_score of v / num edges leaving v
        nsum += solution[v] / outgoing_size(g, v);
      }
      score_new[i] = (damping * nsum) + (1.0 - damping) / numNodes;
      
      // iterate for all 
      double vsum = 0;
      for (const Vertex* v = 0; v!=; v++) {
        if (outgoing_size(g, v) == 0) {
          vsum += (damping * solution[v]) / numNodes;
        }
      }
      score_new[i] += (vsum);
    }
    double global_diff = 0;
    for (int i = 0; i < numNodes; i++) {
      global_diff += abs(score_new[vi] - solution[i]);
    }

    if (global_diff < convergence) {
      converged = true;
    }
    // update
    for(int i = 0; i < numNodes; i++) {
      solution[i] = score_new[i];
    }
  }
  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */

}
