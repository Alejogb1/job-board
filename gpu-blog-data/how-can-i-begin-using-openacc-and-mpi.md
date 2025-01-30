---
title: "How can I begin using OpenACC and MPI in a Fortran program?"
date: "2025-01-30"
id: "how-can-i-begin-using-openacc-and-mpi"
---
Integrating OpenACC directives with MPI within Fortran applications requires careful attention to data management and parallel execution models. Over the past decade, I’ve encountered numerous challenges and best practices in this area, particularly when transitioning from single-node CPU code to a hybrid CPU/GPU cluster environment. The primary complexity arises from ensuring data consistency between the host CPU and the GPU devices while simultaneously coordinating computations across multiple MPI ranks.

OpenACC, at its core, operates through compiler directives embedded within the Fortran code. These directives instruct the compiler to offload computationally intensive loops and data transfers to an accelerator, such as a GPU. Meanwhile, MPI facilitates parallel execution across multiple nodes or processors. The fundamental principle when combining these two paradigms is to handle data parallelism within each MPI process via OpenACC, thereby leveraging both inter-process and intra-process parallelism. The first key step involves correctly partitioning the data across MPI processes, followed by appropriate use of OpenACC for acceleration within each process.

To illustrate, consider a scenario involving a 2D array that needs to be processed. Let us assume the array represents a physical domain, and each point in the array represents a field variable. Without MPI, the entire array would be processed locally, which could take a long time. We could parallelize this locally with OpenACC only. However, we need to scale beyond the limitations of a single GPU. Therefore, we must use MPI. First, we decompose this domain into subdomains, distributing the subdomains across the available MPI processes. Each MPI rank then works on its assigned subdomain and uses OpenACC to accelerate local computations on the GPU if available.

The general approach consists of three essential components. First is to structure the data such that each process is responsible for a specific part of the overall data. Then, we must use MPI to communicate data between the processes to meet any requirements or solve some global problem. Finally, we must use OpenACC within each process to accelerate the computations on its portion of the data. Data consistency must be carefully managed in both the MPI layer between processes and in the OpenACC layer between the CPU and GPU. Explicitly managing data transfers is crucial when a GPU is involved and is one of the most common causes of performance bottlenecks and incorrect computations if not properly handled.

Let us look at three example cases. The first will show a basic example of data decomposition. The second will demonstrate a simple stencil update across an array and demonstrate both intra-process and inter-process data transfers. The third will show a common situation that requires using host-based arrays even with OpenACC for inter-process communication using MPI.

**Example 1: Data Decomposition**

This code illustrates a simple data decomposition for a 2D array across MPI processes. We will use a simple decomposition, which could be improved for performance. This example focuses primarily on managing distributed data. In reality, the subdomains might not all have the same number of elements.

```fortran
program data_decomposition
  use mpi
  implicit none

  integer, parameter :: nx = 1024, ny = 1024
  integer :: rank, size, i, j, istart, iend
  real, dimension(nx, ny) :: global_array, local_array

  call MPI_Init(ierror)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierror)
  call MPI_Comm_size(MPI_COMM_WORLD, size, ierror)

  ! Calculate local array dimensions
  istart = rank * (nx / size) + 1
  iend = (rank + 1) * (nx / size)
  if (rank == size - 1) iend = nx  ! Handle remainder

  ! Allocate and initialize the local array.
  local_array = 0.0

  ! This is a simplified example
  do i = istart, iend
    do j = 1, ny
      local_array(i - istart + 1, j) = real(i + j + 1.0*rank)
      global_array(i,j) = real(i + j + 1.0*rank)
    end do
  end do
  
  ! Print a bit of the local arrays for each rank.
  print *, "Rank: ", rank
  print *, local_array(1:3,1:3)
  
  call MPI_Finalize(ierror)

end program data_decomposition
```

In this example, each rank computes its start and end indices for its portion of the global array. This local array is a smaller version of the global array that can be used in local computations. Note how global_array is never actually accessed, but initialized for testing and verification. Note that in a practical example, both the local and global arrays would likely be declared dynamically with more elegant and appropriate sizes.

**Example 2: Stencil Computation with OpenACC and MPI**

This example extends the prior example to demonstrate a simple 5-point stencil computation, using OpenACC directives within each MPI process. Inter-process communication is required in order to ensure neighboring process data is available, so we are using halo cells for this purpose.

```fortran
program stencil_openacc_mpi
  use mpi
  implicit none

  integer, parameter :: nx = 1024, ny = 1024, halo_width = 1
  integer :: rank, size, i, j, istart, iend, local_nx, local_ny
  real, dimension(:,:), allocatable :: local_array, old_array, halo_west, halo_east
  real, dimension(nx,ny) :: global_array
  integer :: ierror, req1, req2, req3, req4, num_iter
  
  call MPI_Init(ierror)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierror)
  call MPI_Comm_size(MPI_COMM_WORLD, size, ierror)
  
    ! Calculate local array dimensions
  istart = rank * (nx / size) + 1
  iend = (rank + 1) * (nx / size)
  if (rank == size - 1) iend = nx
  
  local_nx = iend - istart + 1
  local_ny = ny
  allocate(local_array(local_nx + 2 * halo_width, local_ny))
  allocate(old_array(local_nx + 2 * halo_width, local_ny))
  allocate(halo_west(halo_width,local_ny))
  allocate(halo_east(halo_width,local_ny))
  
  ! Initialize local_array with values
  do i = 1, local_nx
        do j = 1, local_ny
            local_array(i+halo_width,j) = real(istart+i-1+j + 1.0*rank)
            old_array(i+halo_width,j) = real(istart+i-1+j + 1.0*rank)
            global_array(i+istart-1,j) = real(istart+i-1+j + 1.0*rank)
        end do
  end do

  num_iter = 10
  do i = 1, num_iter
     
  ! Fill the halo regions using MPI sends/receives
  if (rank > 0) then
      halo_west = local_array(halo_width+1:halo_width+halo_width,1:local_ny) 
      call MPI_Isend(halo_west, halo_width * local_ny, MPI_REAL, rank-1, 0, MPI_COMM_WORLD, req1, ierror)
  end if
  if (rank < size - 1) then
      halo_east = local_array(local_nx + halo_width:local_nx+halo_width+halo_width-1,1:local_ny)
      call MPI_Isend(halo_east, halo_width*local_ny, MPI_REAL, rank+1, 1, MPI_COMM_WORLD, req2, ierror)
  end if
  
  ! Receive the halo regions
    if (rank > 0) then
      call MPI_Irecv(local_array(1:halo_width, 1:local_ny), halo_width*local_ny, MPI_REAL, rank-1, 1, MPI_COMM_WORLD, req3, ierror)
  end if
    if (rank < size - 1) then
       call MPI_Irecv(local_array(local_nx+halo_width+halo_width:local_nx+halo_width+2*halo_width-1, 1:local_ny), halo_width*local_ny, MPI_REAL, rank+1, 0, MPI_COMM_WORLD, req4, ierror)
    end if

    if (rank > 0) then
        call MPI_Wait(req3, MPI_STATUS_IGNORE, ierror)
    end if
    if(rank < size-1) then
        call MPI_Wait(req4, MPI_STATUS_IGNORE, ierror)
    end if
   
    ! Perform stencil update on the interior region
    !$acc data copyin(old_array), copyout(local_array)
    !$acc kernels
    !$acc loop independent
    do j = 2, local_ny-1
        !$acc loop independent
        do i = halo_width+1, local_nx+halo_width
          local_array(i,j) = 0.2 * old_array(i,j) + 0.2 * (old_array(i-1,j) + old_array(i+1,j) + old_array(i,j-1) + old_array(i,j+1))
        end do
    end do
    !$acc end kernels
    !$acc end data

    !Copy updated local array to old array for next step
    old_array = local_array
  end do

  if(rank .eq. 0) then
    print *, global_array(1:3,1:3)
  end if
  
  call MPI_Finalize(ierror)

end program stencil_openacc_mpi
```
In this second example, we add explicit halo region exchanges using non-blocking MPI communication. This allows computation to overlap communication, which can improve performance in a strong scaling context. Additionally, we have used OpenACC directives to parallelize the stencil update using the loop-based parallelism within OpenACC. Note that the loop directives within the OpenACC region, namely !`$acc loop independent`, signal to the compiler to generate parallel code where possible.

**Example 3: Host Array Usage with OpenACC and MPI**

In certain situations, data needs to remain on the host even if we are using GPUs. Suppose we need to calculate a global sum of a local reduction, and for simplicity, we will do a reduction to rank 0.

```fortran
program global_reduction
  use mpi
  implicit none

  integer, parameter :: nx = 1024, ny = 1024
  integer :: rank, size, i, j, istart, iend, local_nx, local_ny
  real, dimension(:,:), allocatable :: local_array
  real, dimension(nx,ny) :: global_array
  real :: local_sum, global_sum
  integer :: ierror

  call MPI_Init(ierror)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierror)
  call MPI_Comm_size(MPI_COMM_WORLD, size, ierror)

    ! Calculate local array dimensions
  istart = rank * (nx / size) + 1
  iend = (rank + 1) * (nx / size)
  if (rank == size - 1) iend = nx
  
  local_nx = iend - istart + 1
  local_ny = ny
  allocate(local_array(local_nx, local_ny))
  
  ! Initialize local array with some values
  do i = 1, local_nx
      do j = 1, local_ny
        local_array(i,j) = real(istart+i+j + 1.0*rank)
        global_array(i+istart-1,j) = real(istart+i+j + 1.0*rank)
      end do
  end do

  ! Compute the local sum using OpenACC
  !$acc data copyin(local_array) copyout(local_sum)
  !$acc kernels
    local_sum = 0.0
    !$acc loop reduction(+:local_sum)
    do j = 1, local_ny
        !$acc loop reduction(+:local_sum)
       do i = 1, local_nx
          local_sum = local_sum + local_array(i,j)
       end do
    end do
  !$acc end kernels
  !$acc end data

  ! Reduce to rank 0 via MPI
  if(rank .eq. 0) then
      global_sum = local_sum
    do i = 1, size-1
      call MPI_Recv(local_sum, 1, MPI_REAL, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
      global_sum = global_sum + local_sum
    end do
    print *, "Global Sum:", global_sum
  else
      call MPI_Send(local_sum, 1, MPI_REAL, 0, 0, MPI_COMM_WORLD, ierror)
  end if

  if(rank .eq. 0) then
      print *, "Check array from rank 0"
      print *, global_array(1:3, 1:3)
  end if
  
  call MPI_Finalize(ierror)
end program global_reduction
```
This final example computes a local sum on the GPU using OpenACC, and then passes that sum to the root rank, rank 0, using MPI. Here we do a global sum using a collective reduction, for demonstration and simplicity. The key takeaway here is that while OpenACC is used to compute the local sums on the GPU, these local sums must be accessible to the host CPU for use with MPI, making use of both the data region and the host. 

For further learning, I would strongly advise examining resources covering OpenACC and MPI independently before attempting to combine the two. Specifically, seek comprehensive documentation on OpenACC directives from compiler vendors, such as NVIDIA or PGI. Books or training courses covering MPI are also highly useful. A deep understanding of data layout, process communication, and shared-memory parallelism is critical for effective hybrid programming. Focus on understanding memory models in both MPI and OpenACC, especially when dealing with data movement between host and devices. I recommend starting with simple toy examples, gradually progressing to more complex, real-world scenarios. These skills have consistently proven essential in my own work and will serve any developer well when creating HPC applications.
