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
 * Purpose:     This is a "pass through" VOL connector, which forwards each
 *              VOL callback to an underlying connector.
 *
 *              It is designed as an example VOL connector for developers to
 *              use when creating new connectors, especially connectors that
 *              are outside of the HDF5 library.  As such, it should _NOT_
 *              include _any_ private HDF5 header files.  This connector should
 *              therefore only make public HDF5 API calls and use standard C /
 *              POSIX calls.
 *
 *              Note that the HDF5 error stack must be preserved on code paths
 *              that could be invoked when the underlying VOL connector's
 *              callback can fail.
 *
 */


/* Header files needed */
/* Do NOT include private HDF5 files here! */
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* Public HDF5 headers */
#include "hdf5.h"

/* This connector's private header */
#include "H5VLdrishti.h"
#include "H5VLdrishti_private.h"

/**********/
/* Macros */
/**********/

/* Whether to display log messge when callback is invoked */
/* (Uncomment to enable) */
/* #define ENABLE_DRISHTI_LOGGING */

/* Hack for missing va_copy() in old Visual Studio editions
 * (from H5win2_defs.h - used on VS2012 and earlier)
 */
#if defined(_WIN32) && defined(_MSC_VER) && (_MSC_VER < 1800)
#define va_copy(D,S)      ((D) = (S))
#endif

FILE *drishti_file_g;

/************/
/* Typedefs */
/************/

/* The pass through VOL info object */
typedef struct H5VL_drishti_t {
    hid_t  under_vol_id;        /* ID for underlying VOL connector */
    void   *under_object;       /* Info object for underlying VOL connector */
    MPI_Comm                  comm;
    MPI_Info                  info;
    int                       my_rank;
    int                       num_procs;
} H5VL_drishti_t;

/* The pass through VOL wrapper context */
typedef struct H5VL_drishti_wrap_ctx_t {
    hid_t under_vol_id;         /* VOL ID for under VOL */
    void *under_wrap_ctx;       /* Object wrapping context for under VOL */
    int my_rank;
} H5VL_drishti_wrap_ctx_t;


/********************* */
/* Function prototypes */
/********************* */

/* Helper routines */
static H5VL_drishti_t *H5VL_drishti_new_obj(void *under_obj,
    hid_t under_vol_id);
static herr_t H5VL_drishti_free_obj(H5VL_drishti_t *obj);

/* "Management" callbacks */
static herr_t H5VL_drishti_init(hid_t vipl_id);
static herr_t H5VL_drishti_term(void);

/* VOL info callbacks */
static void *H5VL_drishti_info_copy(const void *info);
static herr_t H5VL_drishti_info_cmp(int *cmp_value, const void *info1, const void *info2);
static herr_t H5VL_drishti_info_free(void *info);
static herr_t H5VL_drishti_info_to_str(const void *info, char **str);
static herr_t H5VL_drishti_str_to_info(const char *str, void **info);

/* VOL object wrap / retrieval callbacks */
static void *H5VL_drishti_get_object(const void *obj);
static herr_t H5VL_drishti_get_wrap_ctx(const void *obj, void **wrap_ctx);
static void *H5VL_drishti_wrap_object(void *obj, H5I_type_t obj_type,
    void *wrap_ctx);
static void *H5VL_drishti_unwrap_object(void *obj);
static herr_t H5VL_drishti_free_wrap_ctx(void *obj);

/* Attribute callbacks */
static void *H5VL_drishti_attr_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t type_id, hid_t space_id, hid_t acpl_id, hid_t aapl_id, hid_t dxpl_id, void **req);
static void *H5VL_drishti_attr_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t aapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_attr_read(void *attr, hid_t mem_type_id, void *buf, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_attr_write(void *attr, hid_t mem_type_id, const void *buf, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_attr_get(void *obj, H5VL_attr_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_attr_specific(void *obj, const H5VL_loc_params_t *loc_params, H5VL_attr_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_attr_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_attr_close(void *attr, hid_t dxpl_id, void **req);

/* Dataset callbacks */
static void *H5VL_drishti_dataset_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t lcpl_id, hid_t type_id, hid_t space_id, hid_t dcpl_id, hid_t dapl_id, hid_t dxpl_id, void **req);
static void *H5VL_drishti_dataset_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t dapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_dataset_read(size_t count, void *dset[],
        hid_t mem_type_id[], hid_t mem_space_id[], hid_t file_space_id[],
        hid_t plist_id, void *buf[], void **req);
static herr_t H5VL_drishti_dataset_write(size_t count, void *dset[],
        hid_t mem_type_id[], hid_t mem_space_id[], hid_t file_space_id[],
        hid_t plist_id, const void *buf[], void **req);
static herr_t H5VL_drishti_dataset_get(void *dset, H5VL_dataset_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_dataset_specific(void *obj, H5VL_dataset_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_dataset_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_dataset_close(void *dset, hid_t dxpl_id, void **req);

/* Datatype callbacks */
static void *H5VL_drishti_datatype_commit(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t type_id, hid_t lcpl_id, hid_t tcpl_id, hid_t tapl_id, hid_t dxpl_id, void **req);
static void *H5VL_drishti_datatype_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t tapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_datatype_get(void *dt, H5VL_datatype_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_datatype_specific(void *obj, H5VL_datatype_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_datatype_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_datatype_close(void *dt, hid_t dxpl_id, void **req);

/* File callbacks */
static void *H5VL_drishti_file_create(const char *name, unsigned flags, hid_t fcpl_id, hid_t fapl_id, hid_t dxpl_id, void **req);
static void *H5VL_drishti_file_open(const char *name, unsigned flags, hid_t fapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_file_get(void *file, H5VL_file_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_file_specific(void *file, H5VL_file_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_file_optional(void *file, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_file_close(void *file, hid_t dxpl_id, void **req);

/* Group callbacks */
static void *H5VL_drishti_group_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id, hid_t dxpl_id, void **req);
static void *H5VL_drishti_group_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t gapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_group_get(void *obj, H5VL_group_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_group_specific(void *obj, H5VL_group_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_group_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_group_close(void *grp, hid_t dxpl_id, void **req);

/* Link callbacks */
static herr_t H5VL_drishti_link_create(H5VL_link_create_args_t *args, void *obj, const H5VL_loc_params_t *loc_params, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_link_copy(void *src_obj, const H5VL_loc_params_t *loc_params1, void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_link_move(void *src_obj, const H5VL_loc_params_t *loc_params1, void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_link_get(void *obj, const H5VL_loc_params_t *loc_params, H5VL_link_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_link_specific(void *obj, const H5VL_loc_params_t *loc_params, H5VL_link_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_link_optional(void *obj, const H5VL_loc_params_t *loc_params, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);

/* Object callbacks */
static void *H5VL_drishti_object_open(void *obj, const H5VL_loc_params_t *loc_params, H5I_type_t *opened_type, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_object_copy(void *src_obj, const H5VL_loc_params_t *src_loc_params, const char *src_name, void *dst_obj, const H5VL_loc_params_t *dst_loc_params, const char *dst_name, hid_t ocpypl_id, hid_t lcpl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_object_get(void *obj, const H5VL_loc_params_t *loc_params, H5VL_object_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_object_specific(void *obj, const H5VL_loc_params_t *loc_params, H5VL_object_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_drishti_object_optional(void *obj, const H5VL_loc_params_t *loc_params, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);

/* Container/connector introspection callbacks */
static herr_t H5VL_drishti_introspect_get_conn_cls(void *obj, H5VL_get_conn_lvl_t lvl, const H5VL_class_t **conn_cls);
static herr_t H5VL_drishti_introspect_get_cap_flags(const void *info, uint64_t *cap_flags);
static herr_t H5VL_drishti_introspect_opt_query(void *obj, H5VL_subclass_t cls, int op_type, uint64_t *flags);

/* Async request callbacks */
static herr_t H5VL_drishti_request_wait(void *req, uint64_t timeout, H5VL_request_status_t *status);
static herr_t H5VL_drishti_request_notify(void *obj, H5VL_request_notify_t cb, void *ctx);
static herr_t H5VL_drishti_request_cancel(void *req, H5VL_request_status_t *status);
static herr_t H5VL_drishti_request_specific(void *req, H5VL_request_specific_args_t *args);
static herr_t H5VL_drishti_request_optional(void *req, H5VL_optional_args_t *args);
static herr_t H5VL_drishti_request_free(void *req);

/* Blob callbacks */
static herr_t H5VL_drishti_blob_put(void *obj, const void *buf, size_t size, void *blob_id, void *ctx);
static herr_t H5VL_drishti_blob_get(void *obj, const void *blob_id, void *buf, size_t size, void *ctx);
static herr_t H5VL_drishti_blob_specific(void *obj, void *blob_id, H5VL_blob_specific_args_t *args);
static herr_t H5VL_drishti_blob_optional(void *obj, void *blob_id, H5VL_optional_args_t *args);

/* Token callbacks */
static herr_t H5VL_drishti_token_cmp(void *obj, const H5O_token_t *token1, const H5O_token_t *token2, int *cmp_value);
static herr_t H5VL_drishti_token_to_str(void *obj, H5I_type_t obj_type, const H5O_token_t *token, char **token_str);
static herr_t H5VL_drishti_token_from_str(void *obj, H5I_type_t obj_type, const char *token_str, H5O_token_t *token);

/* Generic optional callback */
static herr_t H5VL_drishti_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);

/*******************/
/* Local variables */
/*******************/

/* Pass through VOL connector class struct */
static const H5VL_class_t H5VL_drishti_g = {
    H5VL_VERSION,                                       /* VOL class struct version */
    (H5VL_class_value_t)H5VL_DRISHTI_VALUE,        /* value        */
    H5VL_DRISHTI_NAME,                             /* name         */
    H5VL_DRISHTI_VERSION,                          /* connector version */
    0,                                                  /* capability flags */
    H5VL_drishti_init,                         /* initialize   */
    H5VL_drishti_term,                         /* terminate    */
    {                                           /* info_cls */
        sizeof(H5VL_drishti_info_t),           /* size    */
        H5VL_drishti_info_copy,                /* copy    */
        H5VL_drishti_info_cmp,                 /* compare */
        H5VL_drishti_info_free,                /* free    */
        H5VL_drishti_info_to_str,              /* to_str  */
        H5VL_drishti_str_to_info               /* from_str */
    },
    {                                           /* wrap_cls */
        H5VL_drishti_get_object,               /* get_object   */
        H5VL_drishti_get_wrap_ctx,             /* get_wrap_ctx */
        H5VL_drishti_wrap_object,              /* wrap_object  */
        H5VL_drishti_unwrap_object,            /* unwrap_object */
        H5VL_drishti_free_wrap_ctx             /* free_wrap_ctx */
    },
    {                                           /* attribute_cls */
        H5VL_drishti_attr_create,              /* create */
        H5VL_drishti_attr_open,                /* open */
        H5VL_drishti_attr_read,                /* read */
        H5VL_drishti_attr_write,               /* write */
        H5VL_drishti_attr_get,                 /* get */
        H5VL_drishti_attr_specific,            /* specific */
        H5VL_drishti_attr_optional,            /* optional */
        H5VL_drishti_attr_close                /* close */
    },
    {                                           /* dataset_cls */
        H5VL_drishti_dataset_create,           /* create */
        H5VL_drishti_dataset_open,             /* open */
        H5VL_drishti_dataset_read,             /* read */
        H5VL_drishti_dataset_write,            /* write */
        H5VL_drishti_dataset_get,              /* get */
        H5VL_drishti_dataset_specific,         /* specific */
        H5VL_drishti_dataset_optional,         /* optional */
        H5VL_drishti_dataset_close             /* close */
    },
    {                                           /* datatype_cls */
        H5VL_drishti_datatype_commit,          /* commit */
        H5VL_drishti_datatype_open,            /* open */
        H5VL_drishti_datatype_get,             /* get_size */
        H5VL_drishti_datatype_specific,        /* specific */
        H5VL_drishti_datatype_optional,        /* optional */
        H5VL_drishti_datatype_close            /* close */
    },
    {                                           /* file_cls */
        H5VL_drishti_file_create,              /* create */
        H5VL_drishti_file_open,                /* open */
        H5VL_drishti_file_get,                 /* get */
        H5VL_drishti_file_specific,            /* specific */
        H5VL_drishti_file_optional,            /* optional */
        H5VL_drishti_file_close                /* close */
    },
    {                                           /* group_cls */
        H5VL_drishti_group_create,             /* create */
        H5VL_drishti_group_open,               /* open */
        H5VL_drishti_group_get,                /* get */
        H5VL_drishti_group_specific,           /* specific */
        H5VL_drishti_group_optional,           /* optional */
        H5VL_drishti_group_close               /* close */
    },
    {                                           /* link_cls */
        H5VL_drishti_link_create,              /* create */
        H5VL_drishti_link_copy,                /* copy */
        H5VL_drishti_link_move,                /* move */
        H5VL_drishti_link_get,                 /* get */
        H5VL_drishti_link_specific,            /* specific */
        H5VL_drishti_link_optional             /* optional */
    },
    {                                           /* object_cls */
        H5VL_drishti_object_open,              /* open */
        H5VL_drishti_object_copy,              /* copy */
        H5VL_drishti_object_get,               /* get */
        H5VL_drishti_object_specific,          /* specific */
        H5VL_drishti_object_optional           /* optional */
    },
    {                                           /* introspect_cls */
        H5VL_drishti_introspect_get_conn_cls,  /* get_conn_cls */
        H5VL_drishti_introspect_get_cap_flags, /* get_cap_flags */
        H5VL_drishti_introspect_opt_query,     /* opt_query */
    },
    {                                           /* request_cls */
        H5VL_drishti_request_wait,             /* wait */
        H5VL_drishti_request_notify,           /* notify */
        H5VL_drishti_request_cancel,           /* cancel */
        H5VL_drishti_request_specific,         /* specific */
        H5VL_drishti_request_optional,         /* optional */
        H5VL_drishti_request_free              /* free */
    },
    {                                           /* blob_cls */
        H5VL_drishti_blob_put,                 /* put */
        H5VL_drishti_blob_get,                 /* get */
        H5VL_drishti_blob_specific,            /* specific */
        H5VL_drishti_blob_optional             /* optional */
    },
    {                                           /* token_cls */
        H5VL_drishti_token_cmp,                /* cmp */
        H5VL_drishti_token_to_str,             /* to_str */
        H5VL_drishti_token_from_str              /* from_str */
    },
    H5VL_drishti_optional                  /* optional */
};

/* The connector identification number, initialized at runtime */
static hid_t H5VL_DRISHTI_g = H5I_INVALID_HID;

/* Required shim routines, to enable dynamic loading of shared library */
/* The HDF5 library _must_ find routines with these names and signatures
 *      for a shared library that contains a VOL connector to be detected
 *      and loaded at runtime.
 */
H5PL_type_t H5PLget_plugin_type(void) {return H5PL_TYPE_VOL;}
const void *H5PLget_plugin_info(void) {return &H5VL_drishti_g;}


/*-------------------------------------------------------------------------
 * Function:    H5VL__drishti_new_obj
 *
 * Purpose:     Create a new pass through object for an underlying object
 *
 * Return:      Success:    Pointer to the new pass through object
 *              Failure:    NULL
 *
 * Programmer:  Quincey Koziol
 *              Monday, December 3, 2018
 *
 *-------------------------------------------------------------------------
 */
static H5VL_drishti_t *
H5VL_drishti_new_obj(void *under_obj, hid_t under_vol_id)
{
    H5VL_drishti_t *new_obj;

    new_obj = (H5VL_drishti_t *)calloc(1, sizeof(H5VL_drishti_t));
    new_obj->under_object = under_obj;
    new_obj->under_vol_id = under_vol_id;
    H5Iinc_ref(new_obj->under_vol_id);;

    return new_obj;
} /* end H5VL__drishti_new_obj() */


/*-------------------------------------------------------------------------
 * Function:    H5VL__drishti_free_obj
 *
 * Purpose:     Release a pass through object
 *
 * Note:	Take care to preserve the current HDF5 error stack
 *		when calling HDF5 API calls.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 * Programmer:  Quincey Koziol
 *              Monday, December 3, 2018
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_free_obj(H5VL_drishti_t *obj)
{
    hid_t err_id;

    err_id = H5Eget_current_stack();

    H5Idec_ref(obj->under_vol_id);

    H5Eset_current_stack(err_id);

    free(obj);

    return 0;
} /* end H5VL__drishti_free_obj() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_register
 *
 * Purpose:     Register the Dristhi VOL connector and retrieve an ID
 *              for it.
 *
 * Return:      Success:    The ID for the Dristhi VOL connector
 *              Failure:    -1
 *
 * Programmer:  Quincey Koziol
 *              Wednesday, November 28, 2018
 *
 *-------------------------------------------------------------------------
 */
hid_t
H5VL_drishti_register(void)
{
    /* Singleton register the pass-through VOL connector ID */
    if(H5VL_DRISHTI_g < 0)
        H5VL_DRISHTI_g = H5VLregister_connector(&H5VL_drishti_g, H5P_DEFAULT);

    return H5VL_DRISHTI_g;
} /* end H5VL_drishti_register() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_init
 *
 * Purpose:     Initialize this VOL connector, performing any necessary
 *              operations for the connector that will apply to all containers
 *              accessed with the connector.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_init(hid_t vipl_id)
{
#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL INIT\n");
#endif

    /* Shut compiler up about unused parameter */
    (void)vipl_id;

    int flag, my_rank = 0;

    MPI_Initialized(&flag);

    if (flag) {
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    }

    extern char* __progname;

    char fname[128];
    sprintf(fname, "%s.%d.drishti", __progname, my_rank);
    drishti_file_g = fopen(fname, "w");

    return 0;
} /* end H5VL_drishti_init() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_drishti_term
 *
 * Purpose:     Terminate this VOL connector, performing any necessary
 *              operations for the connector that release connector-wide
 *              resources (usually created / initialized with the 'init'
 *              callback).
 *
 * Return:      Success:    0
 *              Failure:    (Can't fail)
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_term(void)
{
#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL TERM\n");
#endif

    /* Reset VOL ID */
    H5VL_DRISHTI_g = H5I_INVALID_HID;

    char *operation;

    /* Write the Drishti traces to a log file */
    while (traces != NULL) {
        if (traces->trace->operation == DRISHTI_TRACE_WRITE) {
            operation = "write";
        } else if (traces->trace->operation == DRISHTI_TRACE_READ) {
            operation = "read";
        } else if (traces->trace->operation == DRISHTI_TRACE_WRITE_METADATA) {
            operation = "write-metadata";
        } else if (traces->trace->operation == DRISHTI_TRACE_READ_METADATA) {
            operation = "read-metadata";
        } else {
            operation = "?";
        }

        fprintf(drishti_file_g, "HDF5;%d;%s;%lu;%lu;%lu;%lld\n", 
            traces->trace->rank,
            operation,
            traces->trace->start,
            traces->trace->end,
            traces->trace->duration,
            traces->trace->offset
        );

        traces = traces->next;
    }    

    fclose(drishti_file_g);

    clear(&traces);

    return 0;
} /* end H5VL_drishti_term() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_drishti_info_copy
 *
 * Purpose:     Duplicate the connector's info object.
 *
 * Returns:     Success:    New connector info object
 *              Failure:    NULL
 *
 *---------------------------------------------------------------------------
 */
static void *
H5VL_drishti_info_copy(const void *_info)
{
    const H5VL_drishti_info_t *info = (const H5VL_drishti_info_t *)_info;
    H5VL_drishti_info_t *new_info;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL INFO Copy\n");
#endif

    /* Allocate new VOL info struct for the pass through connector */
    new_info = (H5VL_drishti_info_t *)calloc(1, sizeof(H5VL_drishti_info_t));

    /* Increment reference count on underlying VOL ID, and copy the VOL info */
    new_info->under_vol_id = info->under_vol_id;
    H5Iinc_ref(new_info->under_vol_id);
    if(info->under_vol_info)
        H5VLcopy_connector_info(new_info->under_vol_id, &(new_info->under_vol_info), info->under_vol_info);

    return new_info;
} /* end H5VL_drishti_info_copy() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_drishti_info_cmp
 *
 * Purpose:     Compare two of the connector's info objects, setting *cmp_value,
 *              following the same rules as strcmp().
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_info_cmp(int *cmp_value, const void *_info1, const void *_info2)
{
    const H5VL_drishti_info_t *info1 = (const H5VL_drishti_info_t *)_info1;
    const H5VL_drishti_info_t *info2 = (const H5VL_drishti_info_t *)_info2;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL INFO Compare\n");
#endif

    /* Sanity checks */
    assert(info1);
    assert(info2);

    /* Initialize comparison value */
    *cmp_value = 0;

    /* Compare under VOL connector classes */
    H5VLcmp_connector_cls(cmp_value, info1->under_vol_id, info2->under_vol_id);
    if(*cmp_value != 0)
        return 0;

    /* Compare under VOL connector info objects */
    H5VLcmp_connector_info(cmp_value, info1->under_vol_id, info1->under_vol_info, info2->under_vol_info);
    if(*cmp_value != 0)
        return 0;

    return 0;
} /* end H5VL_drishti_info_cmp() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_drishti_info_free
 *
 * Purpose:     Release an info object for the connector.
 *
 * Note:	Take care to preserve the current HDF5 error stack
 *		when calling HDF5 API calls.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_info_free(void *_info)
{
    H5VL_drishti_info_t *info = (H5VL_drishti_info_t *)_info;
    hid_t err_id;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL INFO Free\n");
#endif

    err_id = H5Eget_current_stack();

    /* Release underlying VOL ID and info */
    if(info->under_vol_info)
        H5VLfree_connector_info(info->under_vol_id, info->under_vol_info);
    H5Idec_ref(info->under_vol_id);

    H5Eset_current_stack(err_id);

    /* Free pass through info object itself */
    free(info);

    return 0;
} /* end H5VL_drishti_info_free() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_drishti_info_to_str
 *
 * Purpose:     Serialize an info object for this connector into a string
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_info_to_str(const void *_info, char **str)
{
    const H5VL_drishti_info_t *info = (const H5VL_drishti_info_t *)_info;
    H5VL_class_value_t under_value = (H5VL_class_value_t)-1;
    char *under_vol_string = NULL;
    size_t under_vol_str_len = 0;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL INFO To String\n");
#endif

    /* Get value and string for underlying VOL connector */
    H5VLget_value(info->under_vol_id, &under_value);
    H5VLconnector_info_to_str(info->under_vol_info, info->under_vol_id, &under_vol_string);

    /* Determine length of underlying VOL info string */
    if(under_vol_string)
        under_vol_str_len = strlen(under_vol_string);

    /* Allocate space for our info */
    *str = (char *)H5allocate_memory(32 + under_vol_str_len, (hbool_t)0);
    assert(*str);

    /* Encode our info
     * Normally we'd use snprintf() here for a little extra safety, but that
     * call had problems on Windows until recently. So, to be as platform-independent
     * as we can, we're using sprintf() instead.
     */
    sprintf(*str, "under_vol=%u;under_info={%s}", (unsigned)under_value, (under_vol_string ? under_vol_string : ""));

    if(under_vol_string)
        H5free_memory(under_vol_string);
    return 0;
} /* end H5VL_drishti_info_to_str() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_drishti_str_to_info
 *
 * Purpose:     Deserialize a string into an info object for this connector.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_str_to_info(const char *str, void **_info)
{
    H5VL_drishti_info_t *info;
    unsigned under_vol_value;
    const char *under_vol_info_start, *under_vol_info_end;
    hid_t under_vol_id;
    void *under_vol_info = NULL;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL INFO String To Info\n");
#endif

    /* Retrieve the underlying VOL connector value and info */
    sscanf(str, "under_vol=%u;", &under_vol_value);
    under_vol_id = H5VLregister_connector_by_value((H5VL_class_value_t)under_vol_value, H5P_DEFAULT);
    under_vol_info_start = strchr(str, '{');
    under_vol_info_end = strrchr(str, '}');
    assert(under_vol_info_end > under_vol_info_start);
    if(under_vol_info_end != (under_vol_info_start + 1)) {
        char *under_vol_info_str;

        under_vol_info_str = (char *)malloc((size_t)(under_vol_info_end - under_vol_info_start));
        memcpy(under_vol_info_str, under_vol_info_start + 1, (size_t)((under_vol_info_end - under_vol_info_start) - 1));
        *(under_vol_info_str + (under_vol_info_end - under_vol_info_start)) = '\0';

        H5VLconnector_str_to_info(under_vol_info_str, under_vol_id, &under_vol_info);

        free(under_vol_info_str);
    } /* end else */

    /* Allocate new pass-through VOL connector info and set its fields */
    info = (H5VL_drishti_info_t *)calloc(1, sizeof(H5VL_drishti_info_t));
    info->under_vol_id = under_vol_id;
    info->under_vol_info = under_vol_info;

    /* Set return value */
    *_info = info;

    return 0;
} /* end H5VL_drishti_str_to_info() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_drishti_get_object
 *
 * Purpose:     Retrieve the 'data' for a VOL object.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static void *
H5VL_drishti_get_object(const void *obj)
{
    const H5VL_drishti_t *o = (const H5VL_drishti_t *)obj;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL Get object\n");
#endif

    return H5VLget_object(o->under_object, o->under_vol_id);
} /* end H5VL_drishti_get_object() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_drishti_get_wrap_ctx
 *
 * Purpose:     Retrieve a "wrapper context" for an object
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_get_wrap_ctx(const void *obj, void **wrap_ctx)
{
    const H5VL_drishti_t *o = (const H5VL_drishti_t *)obj;
    H5VL_drishti_wrap_ctx_t *new_wrap_ctx;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL WRAP CTX Get\n");
#endif

    /* Allocate new VOL object wrapping context for the pass through connector */
    new_wrap_ctx = (H5VL_drishti_wrap_ctx_t *)calloc(1, sizeof(H5VL_drishti_wrap_ctx_t));

    /* Increment reference count on underlying VOL ID, and copy the VOL info */
    new_wrap_ctx->under_vol_id = o->under_vol_id;
    H5Iinc_ref(new_wrap_ctx->under_vol_id);
    H5VLget_wrap_ctx(o->under_object, o->under_vol_id, &new_wrap_ctx->under_wrap_ctx);

    new_wrap_ctx->my_rank = o->my_rank;

    /* Set wrap context to return */
    *wrap_ctx = new_wrap_ctx;

    return 0;
} /* end H5VL_drishti_get_wrap_ctx() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_drishti_wrap_object
 *
 * Purpose:     Use a "wrapper context" to wrap a data object
 *
 * Return:      Success:    Pointer to wrapped object
 *              Failure:    NULL
 *
 *---------------------------------------------------------------------------
 */
static void *
H5VL_drishti_wrap_object(void *obj, H5I_type_t obj_type, void *_wrap_ctx)
{
    H5VL_drishti_wrap_ctx_t *wrap_ctx = (H5VL_drishti_wrap_ctx_t *)_wrap_ctx;
    H5VL_drishti_t *new_obj;
    void *under;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL WRAP Object\n");
#endif

    /* Wrap the object with the underlying VOL */
    under = H5VLwrap_object(obj, obj_type, wrap_ctx->under_vol_id, wrap_ctx->under_wrap_ctx);
    if(under) {
        new_obj = H5VL_drishti_new_obj(under, wrap_ctx->under_vol_id);
        new_obj->my_rank = wrap_ctx->my_rank;
    }
    else
        new_obj = NULL;

    return new_obj;
} /* end H5VL_drishti_wrap_object() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_drishti_unwrap_object
 *
 * Purpose:     Unwrap a wrapped object, discarding the wrapper, but returning
 *		underlying object.
 *
 * Return:      Success:    Pointer to unwrapped object
 *              Failure:    NULL
 *
 *---------------------------------------------------------------------------
 */
static void *
H5VL_drishti_unwrap_object(void *obj)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    void *under;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL UNWRAP Object\n");
#endif

    /* Unrap the object with the underlying VOL */
    under = H5VLunwrap_object(o->under_object, o->under_vol_id);

    if(under)
        H5VL_drishti_free_obj(o);

    return under;
} /* end H5VL_drishti_unwrap_object() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_drishti_free_wrap_ctx
 *
 * Purpose:     Release a "wrapper context" for an object
 *
 * Note:	Take care to preserve the current HDF5 error stack
 *		when calling HDF5 API calls.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_free_wrap_ctx(void *_wrap_ctx)
{
    H5VL_drishti_wrap_ctx_t *wrap_ctx = (H5VL_drishti_wrap_ctx_t *)_wrap_ctx;
    hid_t err_id;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL WRAP CTX Free\n");
#endif

    err_id = H5Eget_current_stack();

    /* Release underlying VOL ID and wrap context */
    if(wrap_ctx->under_wrap_ctx)
        H5VLfree_wrap_ctx(wrap_ctx->under_wrap_ctx, wrap_ctx->under_vol_id);
    H5Idec_ref(wrap_ctx->under_vol_id);

    H5Eset_current_stack(err_id);

    /* Free pass through wrap context object itself */
    free(wrap_ctx);

    return 0;
} /* end H5VL_drishti_free_wrap_ctx() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_attr_create
 *
 * Purpose:     Creates an attribute on an object.
 *
 * Return:      Success:    Pointer to attribute object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_drishti_attr_create(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t type_id, hid_t space_id, hid_t acpl_id,
    hid_t aapl_id, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *attr;
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    void *under;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL ATTRIBUTE Create\n");
#endif

    under = H5VLattr_create(o->under_object, loc_params, o->under_vol_id, name, type_id, space_id, acpl_id, aapl_id, dxpl_id, req);
    if(under) {
        attr = H5VL_drishti_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_drishti_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        attr = NULL;

    return (void*)attr;
} /* end H5VL_drishti_attr_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_attr_open
 *
 * Purpose:     Opens an attribute on an object.
 *
 * Return:      Success:    Pointer to attribute object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_drishti_attr_open(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t aapl_id, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *attr;
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    void *under;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL ATTRIBUTE Open\n");
#endif

    under = H5VLattr_open(o->under_object, loc_params, o->under_vol_id, name, aapl_id, dxpl_id, req);
    if(under) {
        attr = H5VL_drishti_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_drishti_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        attr = NULL;

    return (void *)attr;
} /* end H5VL_drishti_attr_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_attr_read
 *
 * Purpose:     Reads data from attribute.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_attr_read(void *attr, hid_t mem_type_id, void *buf,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)attr;
    herr_t ret_value;

    long long start, end, duration;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL ATTRIBUTE Read\n");
#endif

    start = drishti_timestamp();

    ret_value = H5VLattr_read(o->under_object, o->under_vol_id, mem_type_id, buf, dxpl_id, req);

    end = drishti_timestamp();

    duration = end - start;

    DRISHTI_trace_info_t *trace = (DRISHTI_trace_info_t *) malloc(sizeof(DRISHTI_trace_info_t));

    trace->rank = o->my_rank;
    trace->offset = -1;
    trace->start = start;
    trace->end = end;
    trace->duration = duration;
    trace->operation = DRISHTI_TRACE_WRITE_METADATA;

    push(&traces, trace);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_attr_read() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_attr_write
 *
 * Purpose:     Writes data to attribute.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_attr_write(void *attr, hid_t mem_type_id, const void *buf,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)attr;
    herr_t ret_value;

    long long start, end, duration;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL ATTRIBUTE Write\n");
#endif

    start = drishti_timestamp();

    ret_value = H5VLattr_write(o->under_object, o->under_vol_id, mem_type_id, buf, dxpl_id, req);

    end = drishti_timestamp();

    duration = end - start;

    DRISHTI_trace_info_t *trace = (DRISHTI_trace_info_t *) malloc(sizeof(DRISHTI_trace_info_t));

    trace->rank = o->my_rank;
    trace->offset = -1;
    trace->start = start;
    trace->end = end;
    trace->duration = duration;
    trace->operation = DRISHTI_TRACE_WRITE_METADATA;

    push(&traces, trace);
    
    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_attr_write() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_attr_get
 *
 * Purpose:     Gets information about an attribute
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_attr_get(void *obj, H5VL_attr_get_args_t *args, hid_t dxpl_id,
    void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL ATTRIBUTE Get\n");
#endif

    ret_value = H5VLattr_get(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_attr_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_attr_specific
 *
 * Purpose:     Specific operation on attribute
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_attr_specific(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_attr_specific_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL ATTRIBUTE Specific\n");
#endif

    ret_value = H5VLattr_specific(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_attr_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_attr_optional
 *
 * Purpose:     Perform a connector-specific operation on an attribute
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_attr_optional(void *obj, H5VL_optional_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL ATTRIBUTE Optional\n");
#endif

    ret_value = H5VLattr_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_attr_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_attr_close
 *
 * Purpose:     Closes an attribute.
 *
 * Return:      Success:    0
 *              Failure:    -1, attr not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_attr_close(void *attr, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)attr;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL ATTRIBUTE Close\n");
#endif

    ret_value = H5VLattr_close(o->under_object, o->under_vol_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    /* Release our wrapper, if underlying attribute was closed */
    if(ret_value >= 0)
        H5VL_drishti_free_obj(o);

    return ret_value;
} /* end H5VL_drishti_attr_close() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_dataset_create
 *
 * Purpose:     Creates a dataset in a container
 *
 * Return:      Success:    Pointer to a dataset object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_drishti_dataset_create(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t lcpl_id, hid_t type_id, hid_t space_id,
    hid_t dcpl_id, hid_t dapl_id, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *dset;
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    void *under;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL DATASET Create\n");
#endif

    under = H5VLdataset_create(o->under_object, loc_params, o->under_vol_id, name, lcpl_id, type_id, space_id, dcpl_id,  dapl_id, dxpl_id, req);
    if(under) {
        dset = H5VL_drishti_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_drishti_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        dset = NULL;

    dset->my_rank = o->my_rank;

    printf("dataset_create rank = %d\n", o->my_rank);

    return (void *)dset;
} /* end H5VL_drishti_dataset_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_dataset_open
 *
 * Purpose:     Opens a dataset in a container
 *
 * Return:      Success:    Pointer to a dataset object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_drishti_dataset_open(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t dapl_id, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *dset;
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    void *under;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL DATASET Open\n");
#endif

    under = H5VLdataset_open(o->under_object, loc_params, o->under_vol_id, name, dapl_id, dxpl_id, req);
    if(under) {
        dset = H5VL_drishti_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_drishti_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        dset = NULL;

    dset->my_rank = o->my_rank;

    printf("dataset_open rank = %d\n", o->my_rank);

    return (void *)dset;
} /* end H5VL_drishti_dataset_open() */



/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_dataset_read
 *
 * Purpose:     Reads data elements from a dataset into a buffer.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_dataset_read(size_t count, void *dset[],
    hid_t mem_type_id[], hid_t mem_space_id[],
    hid_t file_space_id[], hid_t plist_id, void *buf[], void **req)
{
    void *o_arr[count];   /* Array of under objects */
    hid_t under_vol_id;                     /* VOL ID for all objects */
    herr_t ret_value;

    long long start, end, duration;
    //DRISHTI_trace_info_t *drishti[count];
    int ranks[count];

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL DATASET Read\n");
#endif

    /* Populate the array of under objects */
    under_vol_id = ((H5VL_drishti_t *)(dset[0]))->under_vol_id;
    for(size_t u = 0; u < count; u++) {
        o_arr[u] = ((H5VL_drishti_t *)(dset[u]))->under_object;
        assert(under_vol_id == ((H5VL_drishti_t *)(dset[u]))->under_vol_id);

        //drishti[u] = malloc(sizeof(struct DRISHTI_trace_info_t));
        //drishti[u]->rank = ((H5VL_drishti_t *)(dset[u]))->my_rank;
        ranks[u] = ((H5VL_drishti_t *)(dset[u]))->my_rank;
    }

    start = drishti_timestamp();

    ret_value = H5VLdataset_read(count, o_arr, under_vol_id, mem_type_id, mem_space_id, file_space_id, plist_id, buf, req);

    end = drishti_timestamp();

    duration = end - start;

    for(size_t u = 0; u < count; u++) {
        DRISHTI_trace_info_t *trace = (DRISHTI_trace_info_t *) malloc(sizeof(DRISHTI_trace_info_t));

        trace->rank = ranks[u];
        trace->offset = dataset_get_offset(o_arr[u], under_vol_id, H5P_DATASET_XFER_DEFAULT, NULL);
        trace->start = start;
        trace->end = end;
        trace->duration = duration;
        trace->operation = DRISHTI_TRACE_READ;

        push(&traces, trace);
        //drishti[u]->offset = dataset_get_offset(o_arr[u], under_vol_id, H5P_DATASET_XFER_DEFAULT, NULL);

        //drishti[u]->start = start;
        //drishti[u]->end = end;
        //drishti[u]->duration = duration;

        //printf("[rank = %d] %lu -> %lu (%lu microseconds) [offset = %lld]\n", drishti[u]->rank, drishti[u]->start, drishti[u]->end, drishti[u]->duration, drishti[u]->offset);
        //fprintf(drishti_file_g, "%d;read;%lu;%lu;%lu;%lld\n", drishti[u]->rank, drishti[u]->start, drishti[u]->end, drishti[u]->duration, drishti[u]->offset);
    }

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_drishti_dataset_read() */


static hid_t
dataset_get_offset(void *dset, hid_t driver_id, hid_t dxpl_id, void **req)
{
    /* Arguments to VOL callback */
    H5VL_optional_args_t vol_cb_args;
    /* Arguments for optional operation */
    H5VL_native_dataset_optional_args_t dset_opt_args;

    /* Dataset's offset */
    haddr_t dset_offset = HADDR_UNDEF;

    dset_opt_args.get_offset.offset = &dset_offset;
    vol_cb_args.op_type             = H5VL_NATIVE_DATASET_GET_OFFSET;
    vol_cb_args.args                = &dset_opt_args;

    H5VLdataset_optional(dset, driver_id, &vol_cb_args, dxpl_id, req);

    return dset_offset;
}


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_dataset_write
 *
 * Purpose:     Writes data elements from a buffer into a dataset.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_dataset_write(size_t count, void *dset[],
    hid_t mem_type_id[], hid_t mem_space_id[],
    hid_t file_space_id[], hid_t plist_id, const void *buf[], void **req)
{
    void *o_arr[count];   /* Array of under objects */
    hid_t under_vol_id;                     /* VOL ID for all objects */
    herr_t ret_value;

    long long start, end, duration;
    //DRISHTI_trace_info_t *drishti[count];
    int ranks[count];

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL DATASET Write\n");
#endif

    /* Populate the array of under objects */
    under_vol_id = ((H5VL_drishti_t *)(dset[0]))->under_vol_id;

    for(size_t u = 0; u < count; u++) {
        o_arr[u] = ((H5VL_drishti_t *)(dset[u]))->under_object;
        assert(under_vol_id == ((H5VL_drishti_t *)(dset[u]))->under_vol_id);

        //drishti[u] = malloc(sizeof(struct DRISHTI_trace_info_t));
        //drishti[u]->rank = ((H5VL_drishti_t *)(dset[u]))->my_rank;
        ranks[u] = ((H5VL_drishti_t *)(dset[u]))->my_rank;
    }

    start = drishti_timestamp();

    ret_value = H5VLdataset_write(count, o_arr, under_vol_id, mem_type_id, mem_space_id, file_space_id, plist_id, buf, req);

    end = drishti_timestamp();

    duration = end - start;

    for(size_t u = 0; u < count; u++) {
        DRISHTI_trace_info_t *trace = (DRISHTI_trace_info_t *) malloc(sizeof(DRISHTI_trace_info_t));

        trace->rank = ranks[u];
        trace->offset = dataset_get_offset(o_arr[u], under_vol_id, H5P_DATASET_XFER_DEFAULT, NULL);
        trace->start = start;
        trace->end = end;
        trace->duration = duration;
        trace->operation = DRISHTI_TRACE_WRITE;

        push(&traces, trace);

        //drishti[u]->offset = dataset_get_offset(o_arr[u], under_vol_id, H5P_DATASET_XFER_DEFAULT, NULL);

        //drishti[u]->start = start;
        //drishti[u]->end = end;
        //drishti[u]->duration = duration;

        //printf("[rank = %d] %lu -> %lu (%lu microseconds) [offset = %lld]\n", drishti[u]->rank, drishti[u]->start, drishti[u]->end, drishti[u]->duration, drishti[u]->offset);
        //fprintf(drishti_file_g, "%d;write;%lu;%lu;%lu;%lld\n", drishti[u]->rank, drishti[u]->start, drishti[u]->end, drishti[u]->duration, drishti[u]->offset);
    }

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, under_vol_id);

    return ret_value;


/*
    H5VL_drishti_t *o = (H5VL_drishti_t *)dset;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL DATASET Write\n");
#endif

    DRISHTI_trace_info_t *drishti = malloc(sizeof(struct DRISHTI_trace_info_t));

    drishti->offset = dataset_get_offset(o->under_object, o->under_vol_id, NULL, NULL);

    drishti->start = drishti_timestamp();

    ret_value = H5VLdataset_write(o->under_object, o->under_vol_id, mem_type_id, mem_space_id, file_space_id, plist_id, buf, req);

    drishti->end = drishti_timestamp();

    drishti->duration = drishti->end - drishti->start;

    printf("%lu -> %lu (%lu microseconds) [offset = %lld]\n", drishti->start, drishti->end, drishti->duration, drishti->offset);
*/
} /* end H5VL_drishti_dataset_write() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_dataset_get
 *
 * Purpose:     Gets information about a dataset
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_dataset_get(void *dset, H5VL_dataset_get_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)dset;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL DATASET Get\n");
#endif

    ret_value = H5VLdataset_get(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_dataset_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_dataset_specific
 *
 * Purpose:     Specific operation on a dataset
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_dataset_specific(void *obj, H5VL_dataset_specific_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    hid_t under_vol_id;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL H5Dspecific\n");
#endif

    // Save copy of underlying VOL connector ID and prov helper, in case of
    // refresh destroying the current object
    under_vol_id = o->under_vol_id;

    ret_value = H5VLdataset_specific(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_drishti_dataset_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_dataset_optional
 *
 * Purpose:     Perform a connector-specific operation on a dataset
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_dataset_optional(void *obj, H5VL_optional_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL DATASET Optional\n");
#endif

    // TODO: fix this!!
    ret_value = 0; //H5VLdataset_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_dataset_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_dataset_close
 *
 * Purpose:     Closes a dataset.
 *
 * Return:      Success:    0
 *              Failure:    -1, dataset not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_dataset_close(void *dset, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)dset;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL DATASET Close\n");
#endif

    ret_value = H5VLdataset_close(o->under_object, o->under_vol_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    /* Release our wrapper, if underlying dataset was closed */
    if(ret_value >= 0)
        H5VL_drishti_free_obj(o);

    return ret_value;
} /* end H5VL_drishti_dataset_close() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_datatype_commit
 *
 * Purpose:     Commits a datatype inside a container.
 *
 * Return:      Success:    Pointer to datatype object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_drishti_datatype_commit(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t type_id, hid_t lcpl_id, hid_t tcpl_id, hid_t tapl_id,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *dt;
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    void *under;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL DATATYPE Commit\n");
#endif

    under = H5VLdatatype_commit(o->under_object, loc_params, o->under_vol_id, name, type_id, lcpl_id, tcpl_id, tapl_id, dxpl_id, req);
    if(under) {
        dt = H5VL_drishti_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_drishti_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        dt = NULL;

    return (void *)dt;
} /* end H5VL_drishti_datatype_commit() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_datatype_open
 *
 * Purpose:     Opens a named datatype inside a container.
 *
 * Return:      Success:    Pointer to datatype object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_drishti_datatype_open(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t tapl_id, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *dt;
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    void *under;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL DATATYPE Open\n");
#endif

    under = H5VLdatatype_open(o->under_object, loc_params, o->under_vol_id, name, tapl_id, dxpl_id, req);
    if(under) {
        dt = H5VL_drishti_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_drishti_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        dt = NULL;

    return (void *)dt;
} /* end H5VL_drishti_datatype_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_datatype_get
 *
 * Purpose:     Get information about a datatype
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_datatype_get(void *dt, H5VL_datatype_get_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)dt;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL DATATYPE Get\n");
#endif

    ret_value = H5VLdatatype_get(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_datatype_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_datatype_specific
 *
 * Purpose:     Specific operations for datatypes
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_datatype_specific(void *obj, H5VL_datatype_specific_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    hid_t under_vol_id;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL DATATYPE Specific\n");
#endif

    // Save copy of underlying VOL connector ID and prov helper, in case of
    // refresh destroying the current object
    under_vol_id = o->under_vol_id;

    ret_value = H5VLdatatype_specific(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_drishti_datatype_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_datatype_optional
 *
 * Purpose:     Perform a connector-specific operation on a datatype
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_datatype_optional(void *obj, H5VL_optional_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL DATATYPE Optional\n");
#endif

    ret_value = H5VLdatatype_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_datatype_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_datatype_close
 *
 * Purpose:     Closes a datatype.
 *
 * Return:      Success:    0
 *              Failure:    -1, datatype not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_datatype_close(void *dt, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)dt;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL DATATYPE Close\n");
#endif

    assert(o->under_object);

    ret_value = H5VLdatatype_close(o->under_object, o->under_vol_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    /* Release our wrapper, if underlying datatype was closed */
    if(ret_value >= 0)
        H5VL_drishti_free_obj(o);

    return ret_value;
} /* end H5VL_drishti_datatype_close() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_file_create
 *
 * Purpose:     Creates a container using this connector
 *
 * Return:      Success:    Pointer to a file object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_drishti_file_create(const char *name, unsigned flags, hid_t fcpl_id,
    hid_t fapl_id, hid_t dxpl_id, void **req)
{
    H5VL_drishti_info_t *info;
    H5VL_drishti_t *file;
    hid_t under_fapl_id;
    void *under;

    int mpi_initialized;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL FILE Create\n");
#endif

    /* Get copy of our VOL info from FAPL */
    H5Pget_vol_info(fapl_id, (void **)&info);

    /* Make sure we have info about the underlying VOL to be used */
    if (!info)
        return NULL;

    /* Copy the FAPL */
    under_fapl_id = H5Pcopy(fapl_id);

    /* Set the VOL ID and info for the underlying FAPL */
    H5Pset_vol(under_fapl_id, info->under_vol_id, info->under_vol_info);

    /* Open the file with the underlying VOL connector */
    under = H5VLfile_create(name, flags, fcpl_id, under_fapl_id, dxpl_id, req);
    if(under) {
        file = H5VL_drishti_new_obj(under, info->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_drishti_new_obj(*req, info->under_vol_id);
    } /* end if */
    else
        file = NULL;

    file->comm = MPI_COMM_NULL;
    file->info = MPI_INFO_NULL;

    H5Pget_mpi_params(fapl_id, &file->comm, &file->info);

    MPI_Initialized(&mpi_initialized);

    if (file->comm == MPI_COMM_NULL)
        file->comm = MPI_COMM_SELF;

    if (mpi_initialized) {
        MPI_Comm_rank(file->comm, &file->my_rank);
        MPI_Comm_size(file->comm, &file->num_procs);
    } else {
        file->my_rank = 0;
        file->num_procs = 1;
    }

    printf("file_create rank = %d\n", file->my_rank);

    /* Close underlying FAPL */
    H5Pclose(under_fapl_id);

    /* Release copy of our VOL info */
    H5VL_drishti_info_free(info);

    return (void *)file;
} /* end H5VL_drishti_file_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_file_open
 *
 * Purpose:     Opens a container created with this connector
 *
 * Return:      Success:    Pointer to a file object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_drishti_file_open(const char *name, unsigned flags, hid_t fapl_id,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_info_t *info;
    H5VL_drishti_t *file;
    hid_t under_fapl_id;
    void *under;

    int mpi_initialized;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL FILE Open\n");
#endif

    /* Get copy of our VOL info from FAPL */
    H5Pget_vol_info(fapl_id, (void **)&info);

    /* Make sure we have info about the underlying VOL to be used */
    if (!info)
        return NULL;

    /* Copy the FAPL */
    under_fapl_id = H5Pcopy(fapl_id);

    /* Set the VOL ID and info for the underlying FAPL */
    H5Pset_vol(under_fapl_id, info->under_vol_id, info->under_vol_info);

    /* Open the file with the underlying VOL connector */
    under = H5VLfile_open(name, flags, under_fapl_id, dxpl_id, req);
    if(under) {
        file = H5VL_drishti_new_obj(under, info->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_drishti_new_obj(*req, info->under_vol_id);
    } /* end if */
    else
        file = NULL;

    file->comm = MPI_COMM_NULL;
    file->info = MPI_INFO_NULL;

    H5Pget_mpi_params(fapl_id, &file->comm, &file->info);

    MPI_Initialized(&mpi_initialized);

    if (file->comm == MPI_COMM_NULL)
        file->comm = MPI_COMM_SELF;

    if (mpi_initialized) {
        MPI_Comm_rank(file->comm, &file->my_rank);
        MPI_Comm_size(file->comm, &file->num_procs);
    } else {
        file->my_rank = 0;
        file->num_procs = 1;
    }

    printf("file_open rank = %d\n", file->my_rank);

    /* Close underlying FAPL */
    H5Pclose(under_fapl_id);

    /* Release copy of our VOL info */
    H5VL_drishti_info_free(info);

    return (void *)file;
} /* end H5VL_drishti_file_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_file_get
 *
 * Purpose:     Get info about a file
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_file_get(void *file, H5VL_file_get_args_t *args, hid_t dxpl_id,
    void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)file;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL FILE Get\n");
#endif

    ret_value = H5VLfile_get(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_file_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_file_specific
 *
 * Purpose:     Specific operation on file
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_file_specific(void *file, H5VL_file_specific_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)file;
    H5VL_drishti_t *new_o;
    H5VL_file_specific_args_t my_args;
    H5VL_file_specific_args_t *new_args;
    H5VL_drishti_info_t *info;
    hid_t under_vol_id = -1;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL FILE Specific\n");
#endif

    /* Check for 'is accessible' operation */
    if(args->op_type == H5VL_FILE_IS_ACCESSIBLE) {
        /* Make a (shallow) copy of the arguments */
        memcpy(&my_args, args, sizeof(my_args));

        /* Set up the new FAPL for the updated arguments */

        /* Get copy of our VOL info from FAPL */
        H5Pget_vol_info(args->args.is_accessible.fapl_id, (void **)&info);

        /* Make sure we have info about the underlying VOL to be used */
        if (!info)
            return (-1);

        /* Keep the correct underlying VOL ID for later */
        under_vol_id = info->under_vol_id;

        /* Copy the FAPL */
        my_args.args.is_accessible.fapl_id = H5Pcopy(args->args.is_accessible.fapl_id);

        /* Set the VOL ID and info for the underlying FAPL */
        H5Pset_vol(my_args.args.is_accessible.fapl_id, info->under_vol_id, info->under_vol_info);

        /* Set argument pointer to new arguments */
        new_args = &my_args;

        /* Set object pointer for operation */
        new_o = NULL;
    } /* end else-if */
    /* Check for 'delete' operation */
    else if(args->op_type == H5VL_FILE_DELETE) {
        /* Make a (shallow) copy of the arguments */
        memcpy(&my_args, args, sizeof(my_args));

        /* Set up the new FAPL for the updated arguments */

        /* Get copy of our VOL info from FAPL */
        H5Pget_vol_info(args->args.del.fapl_id, (void **)&info);

        /* Make sure we have info about the underlying VOL to be used */
        if (!info)
            return (-1);

        /* Keep the correct underlying VOL ID for later */
        under_vol_id = info->under_vol_id;

        /* Copy the FAPL */
        my_args.args.del.fapl_id = H5Pcopy(args->args.del.fapl_id);

        /* Set the VOL ID and info for the underlying FAPL */
        H5Pset_vol(my_args.args.del.fapl_id, info->under_vol_id, info->under_vol_info);

        /* Set argument pointer to new arguments */
        new_args = &my_args;

        /* Set object pointer for operation */
        new_o = NULL;
    } /* end else-if */
    else {
        /* Keep the correct underlying VOL ID for later */
        under_vol_id = o->under_vol_id;

        /* Set argument pointer to current arguments */
        new_args = args;

        /* Set object pointer for operation */
        new_o = o->under_object;
    } /* end else */

    ret_value = H5VLfile_specific(new_o, under_vol_id, new_args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, under_vol_id);

    /* Check for 'is accessible' operation */
    if(args->op_type == H5VL_FILE_IS_ACCESSIBLE) {
        /* Close underlying FAPL */
        H5Pclose(my_args.args.is_accessible.fapl_id);

        /* Release copy of our VOL info */
        H5VL_drishti_info_free(info);
    } /* end else-if */
    /* Check for 'delete' operation */
    else if(args->op_type == H5VL_FILE_DELETE) {
        /* Close underlying FAPL */
        H5Pclose(my_args.args.del.fapl_id);

        /* Release copy of our VOL info */
        H5VL_drishti_info_free(info);
    } /* end else-if */
    else if(args->op_type == H5VL_FILE_REOPEN) {
        /* Wrap reopened file struct pointer, if we reopened one */
        if(ret_value >= 0 && args->args.reopen.file)
            *args->args.reopen.file = H5VL_drishti_new_obj(*args->args.reopen.file, o->under_vol_id);
    } /* end else */

    return ret_value;
} /* end H5VL_drishti_file_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_file_optional
 *
 * Purpose:     Perform a connector-specific operation on a file
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_file_optional(void *file, H5VL_optional_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)file;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL File Optional\n");
#endif

    ret_value = H5VLfile_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_file_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_file_close
 *
 * Purpose:     Closes a file.
 *
 * Return:      Success:    0
 *              Failure:    -1, file not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_file_close(void *file, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)file;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL FILE Close\n");
#endif

    ret_value = H5VLfile_close(o->under_object, o->under_vol_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    /* Release our wrapper, if underlying file was closed */
    if(ret_value >= 0)
        H5VL_drishti_free_obj(o);

    return ret_value;
} /* end H5VL_drishti_file_close() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_group_create
 *
 * Purpose:     Creates a group inside a container
 *
 * Return:      Success:    Pointer to a group object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_drishti_group_create(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *group;
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    void *under;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL GROUP Create\n");
#endif

    under = H5VLgroup_create(o->under_object, loc_params, o->under_vol_id, name, lcpl_id, gcpl_id,  gapl_id, dxpl_id, req);
    if(under) {
        group = H5VL_drishti_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_drishti_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        group = NULL;

    group->my_rank = o->my_rank;

printf("group_create rank = %d\n", o->my_rank);
    return (void *)group;
} /* end H5VL_drishti_group_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_group_open
 *
 * Purpose:     Opens a group inside a container
 *
 * Return:      Success:    Pointer to a group object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_drishti_group_open(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t gapl_id, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *group;
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    void *under;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL GROUP Open\n");
#endif

    under = H5VLgroup_open(o->under_object, loc_params, o->under_vol_id, name, gapl_id, dxpl_id, req);
    if(under) {
        group = H5VL_drishti_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_drishti_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        group = NULL;

    group->my_rank = o->my_rank;

    printf("group_open rank = %d\n", o->my_rank);

    return (void *)group;
} /* end H5VL_drishti_group_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_group_get
 *
 * Purpose:     Get info about a group
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_group_get(void *obj, H5VL_group_get_args_t *args, hid_t dxpl_id,
    void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL GROUP Get\n");
#endif

    ret_value = H5VLgroup_get(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_group_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_group_specific
 *
 * Purpose:     Specific operation on a group
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_group_specific(void *obj, H5VL_group_specific_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    H5VL_group_specific_args_t my_args;
    H5VL_group_specific_args_t *new_args;
    hid_t under_vol_id;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL GROUP Specific\n");
#endif

    // Save copy of underlying VOL connector ID and prov helper, in case of
    // refresh destroying the current object
    under_vol_id = o->under_vol_id;

    /* Unpack arguments to get at the child file pointer when mounting a file */
    if(args->op_type == H5VL_GROUP_MOUNT) {

        /* Make a (shallow) copy of the arguments */
        memcpy(&my_args, args, sizeof(my_args));

        /* Set the object for the child file */
        my_args.args.mount.child_file = ((H5VL_drishti_t *)args->args.mount.child_file)->under_object;

        /* Point to modified arguments */
        new_args = &my_args;
    } /* end if */
    else
        new_args = args;

    ret_value = H5VLgroup_specific(o->under_object, under_vol_id, new_args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_drishti_group_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_group_optional
 *
 * Purpose:     Perform a connector-specific operation on a group
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_group_optional(void *obj, H5VL_optional_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value = 0;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL GROUP Optional\n");
#endif

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_group_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_group_close
 *
 * Purpose:     Closes a group.
 *
 * Return:      Success:    0
 *              Failure:    -1, group not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_group_close(void *grp, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)grp;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL GROUP Close\n");
#endif

    ret_value = H5VLgroup_close(o->under_object, o->under_vol_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    /* Release our wrapper, if underlying file was closed */
    if(ret_value >= 0)
        H5VL_drishti_free_obj(o);

    return ret_value;
} /* end H5VL_drishti_group_close() */

/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_link_create
 *
 * Purpose:     Creates a hard / soft / UD / external link.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_link_create(H5VL_link_create_args_t *args, void *obj,
    const H5VL_loc_params_t *loc_params, hid_t lcpl_id, hid_t lapl_id,
    hid_t dxpl_id, void **req)
{
    H5VL_link_create_args_t my_args;
    H5VL_link_create_args_t *new_args;
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    hid_t under_vol_id = -1;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL LINK Create\n");
#endif

    /* Try to retrieve the "under" VOL id */
    if(o)
        under_vol_id = o->under_vol_id;

    /* Fix up the link target object for hard link creation */
    if(H5VL_LINK_CREATE_HARD == args->op_type) {
        /* If it's a non-NULL pointer, find the 'under object' and re-set the args */
        if(args->args.hard.curr_obj) {
            /* Make a (shallow) copy of the arguments */
            memcpy(&my_args, args, sizeof(my_args));

            /* Check if we still need the "under" VOL ID */
            if(under_vol_id < 0)
                under_vol_id = ((H5VL_drishti_t *)args->args.hard.curr_obj)->under_vol_id;

            /* Set the object for the link target */
            my_args.args.hard.curr_obj = ((H5VL_drishti_t *)args->args.hard.curr_obj)->under_object;

            /* Set argument pointer to modified parameters */
            new_args = &my_args;
        } /* end if */
        else
            new_args = args;
    } /* end if */
    else
        new_args = args;

    /* Re-issue 'link create' call, possibly using the unwrapped pieces */
    ret_value = H5VLlink_create(new_args, (o ? o->under_object : NULL), loc_params, under_vol_id, lcpl_id, lapl_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_drishti_link_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_link_copy
 *
 * Purpose:     Renames an object within an HDF5 container and copies it to a new
 *              group.  The original name SRC is unlinked from the group graph
 *              and then inserted with the new name DST (which can specify a
 *              new path for the object) as an atomic operation. The names
 *              are interpreted relative to SRC_LOC_ID and
 *              DST_LOC_ID, which are either file IDs or group ID.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_link_copy(void *src_obj, const H5VL_loc_params_t *loc_params1,
    void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t lcpl_id,
    hid_t lapl_id, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o_src = (H5VL_drishti_t *)src_obj;
    H5VL_drishti_t *o_dst = (H5VL_drishti_t *)dst_obj;
    hid_t under_vol_id = -1;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL LINK Copy\n");
#endif

    /* Retrieve the "under" VOL id */
    if(o_src)
        under_vol_id = o_src->under_vol_id;
    else if(o_dst)
        under_vol_id = o_dst->under_vol_id;
    assert(under_vol_id > 0);

    ret_value = H5VLlink_copy((o_src ? o_src->under_object : NULL), loc_params1, (o_dst ? o_dst->under_object : NULL), loc_params2, under_vol_id, lcpl_id, lapl_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_drishti_link_copy() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_link_move
 *
 * Purpose:     Moves a link within an HDF5 file to a new group.  The original
 *              name SRC is unlinked from the group graph
 *              and then inserted with the new name DST (which can specify a
 *              new path for the object) as an atomic operation. The names
 *              are interpreted relative to SRC_LOC_ID and
 *              DST_LOC_ID, which are either file IDs or group ID.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_link_move(void *src_obj, const H5VL_loc_params_t *loc_params1,
    void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t lcpl_id,
    hid_t lapl_id, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o_src = (H5VL_drishti_t *)src_obj;
    H5VL_drishti_t *o_dst = (H5VL_drishti_t *)dst_obj;
    hid_t under_vol_id = -1;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL LINK Move\n");
#endif

    /* Retrieve the "under" VOL id */
    if(o_src)
        under_vol_id = o_src->under_vol_id;
    else if(o_dst)
        under_vol_id = o_dst->under_vol_id;
    assert(under_vol_id > 0);

    ret_value = H5VLlink_move((o_src ? o_src->under_object : NULL), loc_params1, (o_dst ? o_dst->under_object : NULL), loc_params2, under_vol_id, lcpl_id, lapl_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_drishti_link_move() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_link_get
 *
 * Purpose:     Get info about a link
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_link_get(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_link_get_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL LINK Get\n");
#endif

    ret_value = H5VLlink_get(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_link_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_link_specific
 *
 * Purpose:     Specific operation on a link
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_link_specific(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_link_specific_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL LINK Specific\n");
#endif

    ret_value = H5VLlink_specific(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_link_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_link_optional
 *
 * Purpose:     Perform a connector-specific operation on a link
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_link_optional(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL LINK Optional\n");
#endif

    ret_value = H5VLlink_optional(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_link_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_object_open
 *
 * Purpose:     Opens an object inside a container.
 *
 * Return:      Success:    Pointer to object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_drishti_object_open(void *obj, const H5VL_loc_params_t *loc_params,
    H5I_type_t *opened_type, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *new_obj;
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    void *under;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL OBJECT Open\n");
#endif

    under = H5VLobject_open(o->under_object, loc_params, o->under_vol_id, opened_type, dxpl_id, req);
    if(under) {
        new_obj = H5VL_drishti_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_drishti_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        new_obj = NULL;

    return (void *)new_obj;
} /* end H5VL_drishti_object_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_object_copy
 *
 * Purpose:     Copies an object inside a container.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_object_copy(void *src_obj, const H5VL_loc_params_t *src_loc_params,
    const char *src_name, void *dst_obj, const H5VL_loc_params_t *dst_loc_params,
    const char *dst_name, hid_t ocpypl_id, hid_t lcpl_id, hid_t dxpl_id,
    void **req)
{
    H5VL_drishti_t *o_src = (H5VL_drishti_t *)src_obj;
    H5VL_drishti_t *o_dst = (H5VL_drishti_t *)dst_obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL OBJECT Copy\n");
#endif

    ret_value = H5VLobject_copy(o_src->under_object, src_loc_params, src_name, o_dst->under_object, dst_loc_params, dst_name, o_src->under_vol_id, ocpypl_id, lcpl_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o_src->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_object_copy() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_object_get
 *
 * Purpose:     Get info about an object
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_object_get(void *obj, const H5VL_loc_params_t *loc_params, H5VL_object_get_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL OBJECT Get\n");
#endif

    ret_value = H5VLobject_get(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_object_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_object_specific
 *
 * Purpose:     Specific operation on an object
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_object_specific(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_object_specific_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    hid_t under_vol_id;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL OBJECT Specific\n");
#endif

    // Save copy of underlying VOL connector ID and prov helper, in case of
    // refresh destroying the current object
    under_vol_id = o->under_vol_id;

    ret_value = H5VLobject_specific(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_drishti_object_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_object_optional
 *
 * Purpose:     Perform a connector-specific operation for an object
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_object_optional(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL OBJECT Optional\n");
#endif

    ret_value = H5VLobject_optional(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_drishti_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_drishti_object_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_introspect_get_conn_clss
 *
 * Purpose:     Query the connector class.
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_drishti_introspect_get_conn_cls(void *obj, H5VL_get_conn_lvl_t lvl,
    const H5VL_class_t **conn_cls)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL INTROSPECT GetConnCls\n");
#endif

    /* Check for querying this connector's class */
    if(H5VL_GET_CONN_LVL_CURR == lvl) {
        *conn_cls = &H5VL_drishti_g;
        ret_value = 0;
    } /* end if */
    else
        ret_value = H5VLintrospect_get_conn_cls(o->under_object, o->under_vol_id,
            lvl, conn_cls);

    return ret_value;
} /* end H5VL_drishti_introspect_get_conn_cls() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_introspect_get_cap_flags
 *
 * Purpose:     Query the capability flags for this connector and any
 *              underlying connector(s).
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_drishti_introspect_get_cap_flags(const void *_info, uint64_t *cap_flags)
{
    const H5VL_drishti_info_t *info = (const H5VL_drishti_info_t *)_info;
    herr_t                          ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL INTROSPECT GetCapFlags\n");
#endif

    /* Invoke the query on the underlying VOL connector */
    ret_value = H5VLintrospect_get_cap_flags(info->under_vol_info, info->under_vol_id, cap_flags);

    /* Bitwise OR our capability flags in */
    if (ret_value >= 0)
        *cap_flags |= H5VL_drishti_g.cap_flags;

    return ret_value;
} /* end H5VL_drishti_introspect_get_cap_flags() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_introspect_opt_query
 *
 * Purpose:     Query if an optional operation is supported by this connector
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_drishti_introspect_opt_query(void *obj, H5VL_subclass_t cls,
    int op_type, uint64_t *flags)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL INTROSPECT OptQuery\n");
#endif

    ret_value = H5VLintrospect_opt_query(o->under_object, o->under_vol_id, cls,
        op_type, flags);

    return ret_value;
} /* end H5VL_drishti_introspect_opt_query() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_request_wait
 *
 * Purpose:     Wait (with a timeout) for an async operation to complete
 *
 * Note:        Releases the request if the operation has completed and the
 *              connector callback succeeds
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_request_wait(void *obj, uint64_t timeout,
    H5VL_request_status_t *status)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL REQUEST Wait\n");
#endif

    ret_value = H5VLrequest_wait(o->under_object, o->under_vol_id, timeout, status);

    return ret_value;
} /* end H5VL_drishti_request_wait() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_request_notify
 *
 * Purpose:     Registers a user callback to be invoked when an asynchronous
 *              operation completes
 *
 * Note:        Releases the request, if connector callback succeeds
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_request_notify(void *obj, H5VL_request_notify_t cb, void *ctx)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL REQUEST Notify\n");
#endif

    ret_value = H5VLrequest_notify(o->under_object, o->under_vol_id, cb, ctx);

    return ret_value;
} /* end H5VL_drishti_request_notify() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_request_cancel
 *
 * Purpose:     Cancels an asynchronous operation
 *
 * Note:        Releases the request, if connector callback succeeds
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_request_cancel(void *obj, H5VL_request_status_t *status)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL REQUEST Cancel\n");
#endif

    ret_value = H5VLrequest_cancel(o->under_object, o->under_vol_id, status);

    return ret_value;
} /* end H5VL_drishti_request_cancel() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_request_specific
 *
 * Purpose:     Specific operation on a request
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_request_specific(void *obj, H5VL_request_specific_args_t *args)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value = -1;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL REQUEST Specific\n");
#endif

    ret_value = H5VLrequest_specific(o->under_object, o->under_vol_id, args);

    return ret_value;
} /* end H5VL_drishti_request_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_request_optional
 *
 * Purpose:     Perform a connector-specific operation for a request
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_request_optional(void *obj, H5VL_optional_args_t *args)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL REQUEST Optional\n");
#endif

    ret_value = H5VLrequest_optional(o->under_object, o->under_vol_id, args);

    return ret_value;
} /* end H5VL_drishti_request_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_request_free
 *
 * Purpose:     Releases a request, allowing the operation to complete without
 *              application tracking
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_request_free(void *obj)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL REQUEST Free\n");
#endif

    ret_value = H5VLrequest_free(o->under_object, o->under_vol_id);

    if(ret_value >= 0)
        H5VL_drishti_free_obj(o);

    return ret_value;
} /* end H5VL_drishti_request_free() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_blob_put
 *
 * Purpose:     Handles the blob 'put' callback
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_drishti_blob_put(void *obj, const void *buf, size_t size,
    void *blob_id, void *ctx)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL BLOB Put\n");
#endif

    ret_value = H5VLblob_put(o->under_object, o->under_vol_id, buf, size,
        blob_id, ctx);

    return ret_value;
} /* end H5VL_drishti_blob_put() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_blob_get
 *
 * Purpose:     Handles the blob 'get' callback
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_drishti_blob_get(void *obj, const void *blob_id, void *buf,
    size_t size, void *ctx)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL BLOB Get\n");
#endif

    ret_value = H5VLblob_get(o->under_object, o->under_vol_id, blob_id, buf,
        size, ctx);

    return ret_value;
} /* end H5VL_drishti_blob_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_blob_specific
 *
 * Purpose:     Handles the blob 'specific' callback
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_drishti_blob_specific(void *obj, void *blob_id,
    H5VL_blob_specific_args_t *args)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL BLOB Specific\n");
#endif

    ret_value = H5VLblob_specific(o->under_object, o->under_vol_id, blob_id, args);

    return ret_value;
} /* end H5VL_drishti_blob_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_blob_optional
 *
 * Purpose:     Handles the blob 'optional' callback
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_drishti_blob_optional(void *obj, void *blob_id, H5VL_optional_args_t *args)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL BLOB Optional\n");
#endif

    ret_value = H5VLblob_optional(o->under_object, o->under_vol_id, blob_id, args);

    return ret_value;
} /* end H5VL_drishti_blob_optional() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_drishti_token_cmp
 *
 * Purpose:     Compare two of the connector's object tokens, setting
 *              *cmp_value, following the same rules as strcmp().
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_token_cmp(void *obj, const H5O_token_t *token1,
    const H5O_token_t *token2, int *cmp_value)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL TOKEN Compare\n");
#endif

    /* Sanity checks */
    assert(obj);
    assert(token1);
    assert(token2);
    assert(cmp_value);

    ret_value = H5VLtoken_cmp(o->under_object, o->under_vol_id, token1, token2, cmp_value);

    return ret_value;
} /* end H5VL_drishti_token_cmp() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_drishti_token_to_str
 *
 * Purpose:     Serialize the connector's object token into a string.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_token_to_str(void *obj, H5I_type_t obj_type,
    const H5O_token_t *token, char **token_str)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL TOKEN To string\n");
#endif

    /* Sanity checks */
    assert(obj);
    assert(token);
    assert(token_str);

    ret_value = H5VLtoken_to_str(o->under_object, obj_type, o->under_vol_id, token, token_str);

    return ret_value;
} /* end H5VL_drishti_token_to_str() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_drishti_token_from_str
 *
 * Purpose:     Deserialize the connector's object token from a string.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_drishti_token_from_str(void *obj, H5I_type_t obj_type,
    const char *token_str, H5O_token_t *token)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL TOKEN From string\n");
#endif

    /* Sanity checks */
    assert(obj);
    assert(token);
    assert(token_str);

    ret_value = H5VLtoken_from_str(o->under_object, obj_type, o->under_vol_id, token_str, token);

    return ret_value;
} /* end H5VL_drishti_token_from_str() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_drishti_optional
 *
 * Purpose:     Handles the generic 'optional' callback
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_drishti_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_drishti_t *o = (H5VL_drishti_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_DRISHTI_LOGGING
    printf("------- EXT DRISHTI VOL generic Optional\n");
#endif

    ret_value = H5VLoptional(o->under_object, o->under_vol_id, args, dxpl_id, req);

    return ret_value;
} /* end H5VL_drishti_optional() */

struct timespec tp;

unsigned long drishti_timestamp() {
    /*
    struct timeval tv; 
    gettimeofday(&tv, NULL); // get current time
    //long long milliseconds = te.tv_sec*1000LL + te.tv_usec/1000; // calculate milliseconds
    long long microseconds = 1000000 * tv.tv_sec + tv.tv_usec;
    // printf("milliseconds: %lld\n", milliseconds);
    return microseconds;
    */
    //struct timeval tv; 
    //clock_gettime(CLOCK_REALTIME, &tv);
    //long long microseconds = 1000000 * tv.tv_sec + tv.tv_usec;
    //return microseconds;
    clock_gettime(CLOCK_REALTIME, &tp);
    return(((double)tp.tv_sec) + 1.0e-9 * ((double)tp.tv_nsec));
}
