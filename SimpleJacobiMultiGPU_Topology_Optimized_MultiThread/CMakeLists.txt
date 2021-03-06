PROJECT(SimpleJacobiMultiGPU)

set(${PROJECT_NAME}_HEADER
timer.h
)

SET(${PROJECT_NAME}_CUDA_HEADER 
testMultiGPU_Jacobi2D_Decom.cuh
testMultiGPU_Jacobi.cuh
multiGPU_topology_optimize.cuh
)

SET(${PROJECT_NAME}_CUDA_SOURCE
testMultiGPU_Jacobi2D_Decom.cu
testMultiGPU_Jacobi.cu
multiGPU_topology_optimize.cu
)


if(CMAKE_CXX_FLAGS_NO_STD_DEFINED)
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_NO_STD})
endif()

FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
     MESSAGE("OPENMP FOUND")
     SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
     SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  ENDIF()
INCLUDE_DIRECTORIES(SYSTEM ${OpenMP_INCLUDE_PATH})

SOURCE_GROUP(CudaSource FILES ${${PROJECT_NAME}_CUDA_SOURCE})
SOURCE_GROUP(CudaHeader FILES ${${PROJECT_NAME}_CUDA_HEADER})
SOURCE_GROUP(Header FILES ${${PROJECT_NAME}_HEADER})

SET_SOURCE_FILES_PROPERTIES(${${PROJECT_NAME}_CUDA_HEADERS} ${${PROJECT_NAME}_CUDA_KERNELS}
							PROPERTIES HEADER_FILE_ONLY TRUE)

IF(WITH_NSIGHT_DEBUGGING)
	MESSAGE(CUDA DEBUG ON!)
	list(APPEND CUDA_NVCC_FLAGS_DEBUG -G -keep -O0)
ENDIF(WITH_NSIGHT_DEBUGGING)

#find_package(CUDA REQUIRED)
	#option(WITH_LEGACY_CUDA_COMPUTE_2_x "enable building of legacy compute 2.x PTX code")
	#if(WITH_LEGACY_CUDA_COMPUTE_2_x)
		#list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=\"compute_20,sm_20,sm_21\"")
	#endif()
	#list(APPEND CUDA_NVCC_FLAGS
		#"-gencode arch=compute_30,code=\"compute_30,sm_30\"")

MESSAGE(STATUS "CMAKE ${CMAKE_CXX_FLAGS}")
MESSAGE(STATUS "CUDA ${CUDA_NVCC_FLAGS}")
CUDA_ADD_LIBRARY(${PROJECT_NAME} ${${PROJECT_NAME}_CUDA_HEADER} ${${PROJECT_NAME}_CUDA_SOURCE})


