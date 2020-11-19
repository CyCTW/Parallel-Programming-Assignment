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
// #define VERBOSE

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
	bool *vis
	)
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


            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];

                if (distances[outgoing] == NOT_VISITED_MARKER)
                {
                    distances[outgoing] = distances[node] + 1;

                    // int index = new_frontier->count++;
                    local_frontier[local_count] = outgoing;
                    local_count += 1;
                }
            }
            
        }
        #pragma omp critical
        {
            memcpy(new_frontier->vertices + new_frontier->count, local_frontier, local_count*sizeof(int));
            new_frontier->count += local_count;
        }
    }
	for(int i=0; i<new_frontier->count; i++) {
		vis[new_frontier->vertices[i]] = true;
	}
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
	bool* vis = (bool*)malloc(graph->num_nodes * sizeof(bool));
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
		vis[i] = false;
    }
	vis[ROOT_NODE_ID] = true;
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
        top_down_step(graph, frontier, new_frontier, sol->distances, vis);

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
    vertex_set *new_frontier,
    int *distances, 
    bool* vis
    )
{
    int NUM_THREAD = omp_get_num_threads();
    #pragma omp parallel 
    {
        int local_count = 0;
        int *local_frontier = (int*)malloc(sizeof(int)*(g->num_nodes / NUM_THREAD));
        
		#pragma omp for schedule(dynamic, 256*8)
        for(int i = 0; i < g->num_nodes; i++) {

            if ( !vis[i] ) {
                int start_edge = g->incoming_starts[i];
                int end_edge = (i == g->num_nodes - 1)
                            ? g->num_edges
                            : g->incoming_starts[i + 1];

                // iterate all incoming edges
                for(int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    
                    int incoming = g->incoming_edges[neighbor];

                    // if incoming node in frontier
                    if ( vis[incoming] ) {
                        // add vertex i to new frontier
                        distances[i] = distances[incoming] + 1;
                        local_frontier[local_count++] = i;
                        // frontier->vertices[i] = iteration + 1;
                        break;
                    }
                }   
            }

			
        }

        int index = __sync_fetch_and_add(&new_frontier->count, local_count);
		for(int j=0; j<local_count; j++) {
			new_frontier->vertices[index + j] = local_frontier[j];
		}
    }
    for(int i=0; i<new_frontier->count; i++) {
		vis[new_frontier->vertices[i]] = true;
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
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
	bool* vis = (bool*)malloc(graph->num_nodes * sizeof(bool));
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
		vis[i] = false;
    }
    
    // setup frontier with the root node
    frontier->vertices[ROOT_NODE_ID] = 0; // store distance
    frontier->count++;
    sol->distances[ROOT_NODE_ID] = 0;
    vis[ROOT_NODE_ID] = true;

    while (frontier->count > 0)
    {
        frontier->count = 0;
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        bottom_up_step(graph, frontier, new_frontier, sol->distances, vis);
		// swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    }
}


void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    double alpha = 14;
    double beta = 24;

    vertex_set list1;
    vertex_set list2;
	vertex_set_init(&list1, graph->num_nodes);
	vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
	bool* vis = (bool*)malloc(graph->num_nodes * sizeof(bool));
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
		vis[i] = false;
    }
   	vis[ROOT_NODE_ID] = true; 
    // setup frontier with the root node
    frontier->vertices[ROOT_NODE_ID] = 0; // store distance
    frontier->count++;
    sol->distances[ROOT_NODE_ID] = 0;
    int iteration = 0;
    int check_edge_count = 0; // Mf

    int threshold = 10000000;
	int nodes = graph->num_nodes - 1;
	int total_nodes = graph->num_nodes;

	int state = 1;

    while (frontier->count != 0)
    {
		vertex_set_clear(new_frontier);
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif


        if (frontier->count < (total_nodes / beta) ) {
            top_down_step(graph, frontier, new_frontier, sol->distances, vis);
        }
        else {
        	bottom_up_step(graph, frontier, new_frontier, sol->distances, vis);
        }
        iteration++;

		// swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

		nodes -= frontier->count;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    }
}


