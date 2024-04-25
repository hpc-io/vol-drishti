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
 * Purpose:	The public header file for the Dristhi VOL connector.
 */

#ifndef _H5VLdrishti_H
#define _H5VLdrishti_H

/* Public headers needed by this file */
#include "H5VLpublic.h"        /* Virtual Object Layer                 */

/* Identifier for the Dristhi VOL connector */
#define H5VL_DRISHTI	(H5VL_drishti_register())

/* Public characteristics of the Dristhi VOL connector */
#define H5VL_DRISHTI_NAME        "drishti"
#define H5VL_DRISHTI_VALUE       313           /* VOL connector ID */

/* Dristhi VOL connector info */
typedef struct H5VL_drishti_info_t {
    hid_t under_vol_id;         /* VOL ID for under VOL */
    void *under_vol_info;       /* VOL info for under VOL */
} H5VL_drishti_info_t;


typedef struct DRISHTI_trace_info_t {
    unsigned long start, end, duration;
    int operation, rank;
    long long offset;
} DRISHTI_trace_info_t;

typedef struct DRISHTI_trace_t {
    DRISHTI_trace_info_t * trace;

    struct DRISHTI_trace_t * next;
    struct DRISHTI_trace_t * prev;
} DRISHTI_trace_t;


unsigned long drishti_timestamp();

static hid_t
dataset_get_offset(void *dset, hid_t driver_id, hid_t dxpl_id, void **req);

#ifdef __cplusplus
extern "C" {
#endif

/* Technically a private function call, but prototype must be declared here */
extern hid_t H5VL_drishti_register(void);

#ifdef __cplusplus
}
#endif

#endif /* _H5VLdrishti_H */

