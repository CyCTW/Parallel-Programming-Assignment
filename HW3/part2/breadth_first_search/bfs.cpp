#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <unordered_set>
#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define VERBOSE

using namespace std;
void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances, 
    int &check_edge_count)
{
    int local_count; // mf: number of vertices in frontier

    int NUM_THREAD = omp_get_num_threads();

    #pragma omp parallel private(local_count)
    {
        local_count = 0;
        int *local_frontier = (int*)malloc(sizeof(int)*(g->num_nodes / NUM_THREAD));

        #pragma omp for
        for (int i = 0; i < frontier->count; i++)
        {
            // if node i in frontier
            int node = frontier->vertices[i];
            
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->outgoing_starts[node + 1];

            // check_edge_count += (end_edge - start_edge);

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];

                if (distances[outgoing] == NOT_VISITED_MARKER)
                {
                    // #pragma omp critical
                    
                    distances[outgoing] = distances[node] + 1;

                    // int index = new_frontier->count++;
                    local_frontier[local_count] = outgoing;
                    local_count += 1;
                    // frontier->vertices[outgoing] = iteration + 1;
                    // do {
                    // } while( !__sync_bool_compare_and_swap() );
                }
            }
            
        }
        #pragma omp critical
        {
            memcpy(new_frontier->vertices + new_frontier->count, local_frontier, local_count*sizeof(int));
            new_frontier->count += local_count;
        }
    }
    // frontier->count = local_count;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
        // frontier->vertices[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int iteration = 0;
    int check_edge_count = 0;
    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        // frontier->count = 0;
        top_down_step(graph, frontier, new_frontier, sol->distances, check_edge_count);
        // iteration++;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step(Graph g,
    vertex_set *frontier,
    int *distances, 
    int iteration
    )
{
    // int local_count = 0;
    bool judge = false;
    #pragma omp parallel 
    {
        int local_count = 0;
        #pragma omp for schedule(guided)
        for(int i = 0; i < g->num_nodes; i++) {

            if (distances[i] == NOT_VISITED_MARKER) {
                int start_edge = g->incoming_starts[i];
                int end_edge = (i == g->num_nodes - 1)
                            ? g->num_edges
                            : g->incoming_starts[i + 1];

                // iterate all incoming edges
                for(int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    
                    int incoming = g->incoming_edges[neighbor];

                    // if incoming node in frontier
                    if ( distances[incoming] == iteration) {
                        // add vertex i to new frontier
                        distances[i] = distances[incoming] + 1;
                        local_count += 1;
                        // frontier->vertices[i] = iteration + 1;
                        break;
                    }
                }   
            }
        }
        #pragma omp critical 
        {
            frontier->count += local_count;
        }
    }
    

}
void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    vertex_set *frontier = &list1;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
        frontier->vertices[i] = NOT_VISITED_MARKER;
    }
    
    // setup frontier with the root node
    frontier->vertices[ROOT_NODE_ID] = 0; // store distance
    frontier->count++;
    sol->distances[ROOT_NODE_ID] = 0;
    int iteration = 0;

    while (frontier->count != 0)
    {
        frontier->count = 0;
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        bottom_up_step(graph, frontier, sol->distances, iteration);
        iteration++;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    }
}

void top_down_hybrid_step(
    Graph g,
    vertex_set *frontier,
    // vertex_set *new_frontier,
    int *distances, 
    int &check_edge_count,
    int iteration)
{
    int local_count = 0; // mf: number of vertices in frontier

    // int NUM_THREAD = omp_get_num_threads();

    #pragma omp parallel 
    {
        // local_count = 0;
        // int *local_frontier = (int*)malloc(sizeof(int)*(g->num_nodes / NUM_THREAD));

        #pragma omp for reduction(+:local_count, check_edge_count)
        for (int node = 0; node < g->num_nodes; node++)
        {
            // if node i in frontier
            if (frontier->vertices[node] == iteration)
            {
                // int node = frontier->vertices[i];
            
                int start_edge = g->outgoing_starts[node];
                int end_edge = (node == g->num_nodes - 1)
                                ? g->num_edges
                                : g->outgoing_starts[node + 1];

                check_edge_count += (end_edge - start_edge);

                // attempt to add all neighbors to the new frontier
                for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
                {
                    int outgoing = g->outgoing_edges[neighbor];

                    if (distances[outgoing] == NOT_VISITED_MARKER)
                    {
                        // #pragma omp critical
                        
                        distances[outgoing] = distances[node] + 1;

                        // int index = new_frontier->count++;
                        frontier->vertices[outgoing] = iteration + 1;
                        local_count += 1;
                        // frontier->vertices[outgoing] = iteration + 1;
                        // do {
                        // } while( !__sync_bool_compare_and_swap() );
                    }
                }
            }
        }
        
    }
    frontier->count = local_count;
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    double alpha = 0.3;
    double beta = 0.4;

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    vertex_set *frontier = &list1;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
        frontier->vertices[i] = NOT_VISITED_MARKER;
    }
    
    // setup frontier with the root node
    frontier->vertices[ROOT_NODE_ID] = 0; // store distance
    frontier->count++;
    sol->distances[ROOT_NODE_ID] = 0;
    int iteration = 0;
    int check_edge_count = 0; // Mf

    int threshold = 10000000;
    while (frontier->count != 0)
    {
        frontier->count = 0;

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        if (frontier->count < threshold) {
            top_down_hybrid_step(graph, frontier, sol->distances, check_edge_count, iteration);
        }
        else {
            bottom_up_step(graph, frontier, sol->distances, iteration);
        }
        iteration++;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    }
}

