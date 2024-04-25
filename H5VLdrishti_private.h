/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://support.hdfgroup.org/ftp/HDF5/releases.  *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * Purpose:	The private header file for the Dristhi VOL connector.
 */

#ifndef _H5VLdrishti_private_H
#define _H5VLdrishti_private_H

/* Public headers needed by this file */
#include "H5VLdrishti.h"        /* Public header for connector */

/* Private characteristics of the Dristhi VOL connector */
#define H5VL_DRISHTI_VERSION     1

/* The file struct */
typedef struct H5_drishti_file_t {
    char                     *file_name;
    hid_t                     fapl_id;
    hid_t                     fcpl_id;
    MPI_Comm                  comm;
    MPI_Info                  info;
    int                       my_rank;
    int                       num_procs;
} H5_drishti_file_t;

#define DRISHTI_TRACE_WRITE 1
#define DRISHTI_TRACE_READ 2
#define DRISHTI_TRACE_WRITE_METADATA 3
#define DRISHTI_TRACE_READ_METADATA 4

/* H5VL_DRISHTI_DYN_FIDDLE */
    /* No args */

DRISHTI_trace_t *traces = NULL;

void push(DRISHTI_trace_t** head, DRISHTI_trace_info_t *trace)
{
    // 1. allocate node
    DRISHTI_trace_t* new_node = malloc(sizeof(*traces));
 
    // 2. put in the data
    new_node->trace = trace;
 
    // 3. Make next of new node as head and previous as NULL
    new_node->next = (*head);
    new_node->prev = NULL;
 
    // 4. change prev of head node to new node
    if ((*head) != NULL)
        (*head)->prev = new_node;
 
    // 5. move the head to point to the new node
    (*head) = new_node;
}

/*DRISHTI_trace_info_t *pop(DRISHTI_trace_t **head) {
    DRISHTI_trace_t *next = NULL;

    if (*head == NULL) {
        return NULL;
    }

    next = (*head)->next;
    DRISHTI_trace_info_t *element = (*head)->trace;

    free(*head);

    *head = next;

    return element;
}*/

void clear(DRISHTI_trace_t **head) {
    DRISHTI_trace_t *current = (*head);

    while (current != NULL) {
        free(current->trace);

        current = (*head)->next;

        free(head);
    }
}

#ifdef __cplusplus
extern "C" {
#endif


#ifdef __cplusplus
}
#endif

#endif /* _H5VLdrishti_private_H */

