#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *simple_loop(PyObject *self, PyObject *args) {
    // Taken from numpy documentation with slight modifications
    // https://numpy.org/doc/stable/reference/c-api/iterator.html
    PyObject *obj;
    PyArrayObject *array;
    int ok = PyArg_ParseTuple(args, "O", &obj);
    if (!ok)
        return NULL;

    if (!PyArray_Check(obj))
        Py_RETURN_NONE;
    else
        array = (PyArrayObject *)obj;

    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    char** dataptr;
    npy_intp* strideptr, *innersizeptr;

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(array) == 0) {
        Py_RETURN_NONE;
    }

  /*
     * Create and use an iterator to count the nonzeros.
     *   flag NPY_ITER_READONLY
     *     - The array is never written to.
     *   flag NPY_ITER_EXTERNAL_LOOP
     *     - Inner loop is done outside the iterator for efficiency.
     *   flag NPY_ITER_NPY_ITER_REFS_OK
     *     - Reference types are acceptable.
     *   order NPY_KEEPORDER
     *     - Visit elements in memory order, regardless of strides.
     *       This is good for performance when the specific order
     *       elements are visited is unimportant.
     *   casting NPY_NO_CASTING
     *     - No casting is required for this operation.
     */
    iter = NpyIter_New(array, NPY_ITER_READONLY|
                             NPY_ITER_EXTERNAL_LOOP|
                             NPY_ITER_REFS_OK,
                        NPY_KEEPORDER, NPY_NO_CASTING,
                        NULL);
    if (iter == NULL) {
        Py_RETURN_NONE;
    }

    /*
     * The iternext function gets stored in a local variable
     * so it can be called repeatedly in an efficient manner.
     */
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        Py_RETURN_NONE;
    }
    /* The location of the data pointer which the iterator may update */
    dataptr = NpyIter_GetDataPtrArray(iter);
    /* The location of the stride which the iterator may update */
    strideptr = NpyIter_GetInnerStrideArray(iter);
    /* The location of the inner loop size which the iterator may update */
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    do {
        printf("Starting inner loop\n");        
        /* Get the inner loop data/stride/count values */
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;

        /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
        while (count--) {
            printf("Iterating inner elements\n");
            data += stride;
        }
        printf("Leaving inner loop\n");

        /* Increment the iterator to the next inner loop */
    } while(iternext(iter));

    NpyIter_Deallocate(iter);

    Py_RETURN_NONE;
};
    

static PyMethodDef methods[] = {
    {"simple_loop", simple_loop, METH_VARARGS, "Showcases simple iteration"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef npyitersmodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "npyiters",
    .m_doc = PyDoc_STR("Examples on how to use the NumPy C iteration API"),
    .m_methods = methods,
};


PyMODINIT_FUNC
PyInit_npyiters(void) {
    import_array();
    return PyModule_Create(&npyitersmodule);
}
