#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

int PrintMultiIndex(PyArrayObject *arr) {
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    char **dataptr;
    iter = NpyIter_New(
        arr, NPY_ITER_READONLY | NPY_ITER_MULTI_INDEX | NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (iter == NULL) {
        return -1;
    }
    if (NpyIter_GetNDim(iter) != 2) {
        NpyIter_Deallocate(iter);
        PyErr_SetString(PyExc_ValueError, "Array must be 2-D");
        return -1;
    }
    if (NpyIter_GetIterSize(iter) != 0) {
        npy_intp *multi_index;
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            return -1;
        }
        NpyIter_GetMultiIndexFunc *get_multi_index =
            NpyIter_GetGetMultiIndex(iter, NULL);
        if (get_multi_index == NULL) {
            NpyIter_Deallocate(iter);
            return -1;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        do {
            get_multi_index(iter, multi_index);
            printf("multi_index is [%" NPY_INTP_FMT ", %" NPY_INTP_FMT "]\n",
                   multi_index[0], multi_index[1]);
        } while (iternext(iter));
    }
    if (!NpyIter_Deallocate(iter)) {
        return -1;
    }
    return 0;
}

PyObject *print_2d(PyObject *self, PyObject *args) {
    PyObject *obj;
    PyArrayObject *array;
    int ok = PyArg_ParseTuple(args, "O", &obj);
    if (!ok)
        return NULL;

    if (!PyArray_Check(obj)) {
      PyErr_SetString(PyExc_TypeError, "Expected numpy array");
      return NULL;
    } else {
      int ret = PrintMultiIndex((PyArrayObject *)obj);
      if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Something unexpected happended");
        return NULL;
      }
    }

    Py_RETURN_NONE;
};
    

static PyMethodDef methods[] = {
    {"print_2d", print_2d, METH_VARARGS, "Prints 2D Multi Index position"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef npyitersmodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "npyiters",
    .m_methods = methods,
};


PyMODINIT_FUNC
PyInit_npyiters(void) {
    import_array();
    return PyModuleDef_Init(&npyitersmodule);
}
