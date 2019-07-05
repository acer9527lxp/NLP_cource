from collections import  deque
import random
import time
import numpy as np

def Karatsuba_Multiplication(x,y,base=10):
    """ :return the result of x multiplication of x and y  use Karatsuba Multiplication
    :param x  int:
    :param y int :
    :return:
    """

    if (len(str(x)) == 1 or len(str(y)) == 1):
        return x*y
    n = max(len(str(x)),len(str(y)))

    half = int(n//2)
    x_h = int(x//(base**(half)))
    x_l = int(x%(base**(half)))

    y_h = int(y // (base ** (half)))
    y_l = int(y % (base ** (half)))

    print(f"{x_h},{x_l},{y_h},{y_l}")

    ac=Karatsuba_Multiplication(x_h , y_h)
    bd=Karatsuba_Multiplication(x_l , y_l)
    z=Karatsuba_Multiplication((x_h + x_l),(y_h + y_l)) - ac - bd
    return ac*(np.power(base,half*2)) +z*(base**(half)) + bd


class solution(object):
    """
    MergeSort() implication MergeSort
    MergeSortR() implication MergeSort by random select the pivot
    Qsort()
    SelectSort()
    SelectSortR()
    BubbleSort()
    Insert Sort()
    Select Sort()
    """
    def __init__(self,list):
        self.list = list

    def merge(left, right):
        res = []
        while left and right:
            if left[0] < right[0]:
                res.append(left.pop(0))
            else:
                res.append(right.pop(0))
        res = res + left + right
        return res

    def MergeSort(self,array):
        """ O(n*log(n))
        :param list:
        :return:
        """
        list = self.list.copy()

        if len(list)<= 1:
            return list
        half = int(len(list) // 2)
        print("half= ",half)

        left_sorted_list = self.MergeSort(list[:half])
        right_sorted_list = self.MergeSort(list[half:])

        return self.merge(left_sorted_list, right_sorted_list)

    def BubbleSort(self):
        """ O(n**2)
        :return:
        """
        list = self.list.copy()
        if len(list) <= 1:
            return  list

        for i in range(len(list)):
            for j in range(len(list)-i-1):
                if list[j]> list[j+1]:
                    temp = list[j]
                    list[j] = list[j+1]
                    list[j+1]=temp
        return list



def  binary_search(list,item):
    """ find item from list，list must be sorted!!!
    :param list: find from
    :param item: be find
    :return: none if  iten not in list else the pos
    """
    low = 0
    high = len(list) - 1
    while low <=high:
        mid = int((low + high) / 2)
        guss  =  list[mid]

        if guss == item:
            print("Yes ,find it : ")
            print(f"{item} at the No.{mid} elements of the list")
            return mid
        if guss > item:
            high = mid -1
        else:
            low = mid +1
    print(f"{item} is not find from list.")
    return None

def findsmallest(arr):
    """find the smallest element
    :param arr:
    :return: smallest element and the index
    """

    smallest = arr[0]
    smallest_index = 0

    for i in range(1,len(arr)-1):
        if arr[i]<smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest,smallest_index

def selectionSort(arr):
    """
    :param arr:
    :return:
    """
    newarr = []
    for i in range(len(arr)):
        _,smallest_index = findsmallest(arr)

        newarr.append(arr.pop(smallest_index))

    return newarr

def fact(x):
    """
    :param x:
    :return:
    """
    if x == 1:
        return x
    else:
        return  x*fact((x-1))

def sum_loop(list):
    """ return the sum of all elements in the list
    :param list:
    :return:
    """
    sum = 0
    for iten in list:
        sum +=iten

    return  sum


def sum_recursive(list):
    """
    :param list:
    :return:
    """
    if len(list)<=1:
        return list[0]
    else:
        return list[0] +sum_recursive(list[1:])

def cuount_recursive(list):
    """
    :param list:
    :return:
    """
    if list==[]:
        return 0
    else:
        return 1+cuount_recursive(list[1:])


def max_recursive(list):
    """

    :param list:
    :return:
    """
    if len(list) == 0:
        return None
    elif len(list) == 1:
        return list[0]
    else:
         max_value  = max_recursive(list[1:])
    return list[0] if list[0]>max_value else max_value

def qsort(list):
    """
    :param list:
    :return:
    """

    if len(list)<2:
        return list
    else:
        cur = list[0]
        left = [item for item in list[1:]  if item <=cur ]
        right = [item for item in list[1:] if item > cur]

    return qsort(left)+[cur]+qsort(right)


def count_inversions_loop(list):
    """count the inversions in the list
    :param list:
    :return:
    """
    invs_num = 0

    if len(list)<=1:
        return 0
    else:
        for i in range(len(list)-1):
            for j in range(i,len(list)):
                if list[i] > list[j]:
                    invs_num +=1
    return invs_num

def person_is_saller(name):
    """ :return true if name['-1] == 'm ' else false
    :param name:
    :return:
    """
    return name[-1] == 'm'


def BFS(graph,name):
    """
    :param graph:
    :return:
    """
    search_queue = deque()
    search_queue += graph[name]
    searched_list = {}

    while search_queue:
        person = search_queue.popleft()
        if not searched_list.get(person):
            searched_list[person] = 1
            if person_is_saller(person):
                print(person + " is a mango seller!")
                break
            else:
                search_queue +=graph[person]
    print("have searhed person list: ",searched_list.keys())
    return False

def get_second_heighest(a):
    mid = len(a)
    for _ in range(mid.bit_length()):
        mid, add = divmod(mid, 2)
        add += mid
        for i in range(mid):
            if a[i] < a[i + add]:
                a[i], a[i + add] = a[i + add], a[i]
    if len(a) >= 2:
        return a[0],a[1]
    return None

def findsecondbigest(list):
    """
    :param list:
    :return:
    """
    first = 0
    second = 0
    for i in range(len(list)):
        if list[i] >first:
            second = first
            first = list[i]
        else:
            if list[i] < first and list[i] > second:
                second = list[i]
    return first,second

graph = {}
graph["you"] = ["alice", "bob", "claire"]
graph["bob"] = ["anuj", "peggy"]
graph["alice"] = ["peggy"]
graph["claire"] = ["thom", "jonny"]
graph["anuj"] = []
graph["peggy"] = []
graph["thom"] = []
graph["jonny"] = []


def partition(list, beg, end, pivot_flag='F'):
    """
    :param list:
    :param beg:
    :param end:
    :param pivot_flag: the pivot element position
    :return:
    """

    print("subarray is : ",list[beg:end])
    print("lenght: ",len(list[beg:end]) - 1)
    if pivot_flag == 'F':
        pivot_index =beg
    elif pivot_flag == 'L':
        pivot_index =beg
    else:
        pivot_index = random.randint(beg, end)

    comparisons = 0

    pivot = list[pivot_index]
    left = pivot_index +1
    right = end-1

    while True:
        while left <= right and list[left] < pivot:
            comparisons +=1
            left +=1
        while right >= left and list[right] > pivot:
            comparisons += 1
            right -=1
        # 当两个while循环结束的时候，找到的两个元素。左指针指向的元素大于pivot， 右指针指向的元素大于等于pivot。
        # 交换两个指针指向的值
        if left >= right:
            break
        else:
            list[left], list[right] = list[right], list[left]
    # 退出循环后，right在left的左边，此时right指向的值是小于pivot的，所有将array[pivot_index]和array[right]交换
    list[pivot_index], list[right] = list[right], list[pivot_index]

    print("comparisons times is: ", comparisons)

    return right,len(list[beg:end]) - 1


def RandPartition(list,beg,end):
    """ base on random select the pivot
    :param list:
    :param beg:
    :param end:
    :return:
    """
    pivot_index = random.randint(beg, end)
    print("the pivot_index is: ",pivot_index)

    # if the pivot_index is not the beg then swap the list[beg], list[pivot_index]
    if pivot_index != beg:
        list[beg], list[pivot_index] = list[pivot_index], list[beg]

    pivot = list[pivot_index]
    left = pivot_index +1
    right = end-1

    while True:
        while left <= right and list[left] < pivot:
            left +=1
        while right >= left and list[right] > pivot:
            right -=1
        # 当两个while循环结束的时候，找到的两个元素。左指针指向的元素大于pivot， 右指针指向的元素大于等于pivot。
        # 交换两个指针指向的值
        if left >= right:
            break
        else:
            list[left], list[right] = list[right], list[left]
    # 退出循环后，right在left的左边，此时right指向的值是小于pivot的，所有将array[pivot_index]和array[right]交换
    list[pivot_index], list[right] = list[right], list[pivot_index]

    return right



def quicksort_inplace(array, beg, end):
    left_comparisons,right_comparisons, total_comparisons = 0, 0, 0
    if beg<end:

        pivot,total_comparisons = partition(array, beg, end, 'F')
        array,left_comparisons = quicksort_inplace(array,beg, pivot)
        array, right_comparisons = quicksort_inplace(array, pivot+1, end)

    return array,total_comparisons + left_comparisons + right_comparisons


def RSelect(array, left, right, i):
    """
    :param array:
    :param left:
    :param right:
    :param i:
    :return:
    """
    if left==right:
        return  array[left]
    else:
        pivot_index = RandPartition(array, left, right)

        if pivot_index == i:
            print("get it: ",array[pivot_index])
        else:
            if pivot_index > i :
                return RSelect(array, left, pivot_index-1, i)
            else:
                return RSelect(array, pivot_index + 1, right, i)

def CountSplitInv(left,right):
    """count the  split inversions
    :param left:
    :param right:
    :return: (combine array1 and array2,the split inversions between array1 and array2)
    """
    i, j,k, splitInv = 0,0,0,0
    left_len = len(left)
    right_len = len(right)
    result = []
    while i < left_len and j < right_len:
        if (left[i] <= right[j]):
            result.append(left[i])
            i +=1
        else :
            result.append(right[j])
            j +=1
            splitInv += (left_len - i)
    # left is complete
    result +=right[j:]
    # right is complete
    result += left[i:]
    return (result, splitInv)


def SortandCountInv(array):
    """ sorted array B with the same integers, and
        the number of inversions of A
    :param array:
    :return: 
    """
    if len(array) <= 1:
        return (array,0)
    else:
        mid = len(array) // 2
        (left,leftInv) =SortandCountInv(array[:mid])
        (right, rightInv) = SortandCountInv(array[mid:])
        (sortedarray,splitInv) = CountSplitInv(left,right)
    return (sortedarray,leftInv+rightInv+splitInv)

def test():
    input_array_1 = []  # 0
    input_array_2 = [1]  # 0
    input_array_3 = [1, 5]  # 0
    input_array_4 = [4, 1]  # 1
    input_array_5 = [4, 1, 2, 3, 9]  # 3
    input_array_6 = [4, 1, 3, 2, 9, 5]  # 5
    input_array_7 = [4, 1, 3, 2, 9, 1]  # 8

    assert SortandCountInv(input_array_1) == ([], 0)
    assert SortandCountInv(input_array_2) == ([1], 0)
    assert SortandCountInv(input_array_3) == ([1,5], 0)
    assert SortandCountInv(input_array_4) == ([1,4], 1)
    assert SortandCountInv(input_array_5) == ([1,2,3,4,9], 3)
    assert SortandCountInv(input_array_6) == ([1,2,3,4,5,9], 5)
    assert SortandCountInv(input_array_7) == ([1,1,2,3,4,9], 8)

    print("test pass")


def read_data(data_file):
    """ read data from file
    :param data_file:
    :return: data in array
    """

    data =[]
    with open(data_file) as f:
        contents = f.readlines()
        for line in contents:
            line = line.split('\n')[0]
            data.append(int(line))

    return data

def Partition2(A,l,r,CT):
    p=A[l]
    i=l+1
    for j in np.arange(l+1,r+1,1):
        if A[j]<p:
            A[np.array([i,j])]=A[np.array([j,i])]
            i=i+1
        CT=CT+1
    A[np.array([l,i-1])]=A[np.array([i-1,l])]
    return (A,i-1,CT)

def CT_of_QS_of_first(A,l,r,CT):
    if l>=r:
        return (A,CT)
    else:
        #下面的代码将首元素下标作为支点指标
        i=l
        A[np.array([l,i])]=A[np.array([i,l])]
        (A,j,CT1)=Partition2(A,l,r,CT)
        (A,CT2)=CT_of_QS_of_first(A,l,j-1,CT1)
        (A,CT3)=CT_of_QS_of_first(A,j+1,r,CT2)
        return (A,CT3)
def CT_of_QS_of_last(A,l,r,CT):
    if l>=r:
        return (A,CT)
    else:
        #下面的代码将最后元素下标作为支点指标
        i=r
        A[np.array([l,i])]=A[np.array([i,l])]
        (A,j,CT1)=Partition2(A,l,r,CT)
        (A,CT2)=CT_of_QS_of_last(A,l,j-1,CT1)
        (A,CT3)=CT_of_QS_of_last(A,j+1,r,CT2)
        return (A,CT3)

if __name__ == "__main__":
    my_list = [1, 3, 5, 7, 9, 11, 12, 15]
    start_time = time.time()
    # print(binary_search(my_list,3))
    # print(binary_search(my_list,10))

    import math
    # print(math.log2(10000000000))
    # # print(math.log2(256))

    # print(selectionSort([5, 3, 6, 2, 10]))

    # print(fact(5))

    # print(sum_loop(my_list))
    # print(sum_recursive(my_list))

    # print(cuount_recursive(my_list))
    # print(max_recursive(my_list))
    # print(qsort([5,4,3,2,1,9]))
    # print(count_inversions_loop([1,3,5,2,4,6]))
    # BFS(graph, 'you')

    # print(quicksort_inplace([3,8,2,5,1,4,7,6],0,8))
    # print(RSelect([3,8,2,5,1,4,7,6],0,8,3))
    #     # cpu_time = time.time()-start_time
    #     # print(f"total CPU TIME is: {cpu_time}")

    # time_start = time.time()
    # print(Karatsuba_Multiplication(5678,1234))
    # print(Karatsuba_Multiplication(3141592653589793238462643383279502884197169399375105820974944592,
    #                                2718281828459045235360287471352662497757247093699959574966967627))
    # total_time = time.time() -time_start
    # print(f"time useing :{total_time}")
    bubblesort=solution([3,8,2,5,1,4,7,6])
    print(bubblesort.list)
    print(bubblesort.MergeSort())
    # print(MergeSort([3,8,2,5,1,4,7,6]))
    # print(bubblesort.BubbleSort())
    # print(findsecondbigest([3,8,2,5,1,4,7,6]))
    # print(get_second_heighest([3,8,2,5,1,4,7,6]))
    # print(get_second_heighest([3, 101, 2, 100]))
    # print(SortandCountInv([3,8,2,5,1,4,7,6],0,7))

    #homework1
    # data =[]
    # with open("../data/IntegerArray.txt") as f:
    #     contents = f.readlines()
    #     for line in contents:
    #         line = line.split('\n')[0]
    #         data.append(int(line))
    #
    # print(data[:10])
    #
    # print(SortandCountInv(data))
    # print(f"total CPU TIME is: {time.time() - start_time}")

    # print(CountSplitInv([2,3,6],[1,4,7]))
    # print(SortandCountInv())
    # test()
    # print(SortandCountInv([4, 1, 3, 2, 9, 1]))

    # homework2
    # data_file = "../data/QuickSort.txt"
    # array_10000 = read_data(data_file)
    # print(quicksort_inplace(array_10000, 0, 10000))
    # print(quicksort_inplace([3,8,2,5,1,4,7,6], 0, 8))
    # print(quicksort_inplace([3,8,2,5,1,4,7,6,9,11], 0, 10))

    # A = np.loadtxt('../data/QuickSort.txt',delimiter='\n')
    # print("####", A[np.array([1,1])])
    # print(CT_of_QS_of_first(A, 0, (len(A) - 1), 0))
    # print(CT_of_QS_of_last(A, 0, (len(A) - 1), 0))