def main():

    x=0
    h=0
    f = open('file.txt', 'r')
    i = 1
    mystring = []
    mystring2 = []
    list1=[]
    contents = f.read()
    mystring.append(contents)

    for line in contents:
        for c in line.split():
            mystring2.append(c)
    print(mystring2)
    for index in range(len(mystring2)):
      # The current index value
            indexw=index
            if mystring2[indexw]=='/':
                continue
            else:
                print(mystring2[indexw])
                list1.append(mystring2[indexw])

        #for right value with respect to that index
            indexu=index+1
            if mystring2[indexu]=='/':
                print("Null")
                list1.append("Null")
            else:
                print(mystring2[indexu])
                list1.append(mystring2[indexu])
                index=indexu-1
        #for down value with respect to that index

            indexc = index + 9
            if mystring2[indexc] == '/':
                list1.append("Null")
                print("Null")
            else:
                print(mystring2[indexc])
                list1.append(mystring2[indexc])
                index=indexc-9

        #for upward value with respect to that index
            indext = index - 9
            if mystring2[indext] == '/':
                 list1.append("Null")
                 print("Null")
            else:
                 print(mystring2[indext])
                 list1.append(mystring2[indext])
                 index=indext+9
         #for left value with respect to that index
            indexf = index - 1
            if mystring2[indexf] == '/':
                list1.append("Null")
                print("Null")
            else:
               print(mystring2[indexf])
               list1.append(mystring2[indexf])
               index=indexf+1
            print(list1)
            list1.clear()


# def y():
#     dest_array = [360, 360, 270, 90, 180, 360]
#     f = open('file.txt', 'r')
#     cont = f.read()
#     for word in f:
#         print(word)
#     for index in range(len(dest_array)):
#         if dest_array[index] == '360':
#             w = f.tell()
#             x = f.seek(w + 1)
#             print(x)
#         elif dest_array[index] == '270':
#             s = f.tell()
#             f.seek(s + 7)
#         elif dest_array[index] == '90':
#             g = f.tell()
#             f.seek(g - 7)
#         elif dest_array[index] == "180":
#             b = f.tell()
#             f.seek(b - 1)
#         break

            # tree = Tree()
            # tree.add(3)
            # tree.add(5)
            # tree._printTree()

# class Node:
#     def __init__(self, val):
#         self.l = None
#         self.r = None
#         self.v = val
#
# class Tree:
#         def __init__(self):
#             self.root = None
#
#
#         def getRoot(self):
#             return self.root
#
#         def add(self, val):
#             if (self.root == None):
#                 self.root = Node(val)
#             else:
#                 self.add(val, self.root)
#
#         def _printTree(self, node):
#             if (node != None):
#                 self._printTree(node.l)
#                 print(str(node.v) + ' ')
#                 self._printTree(node.r)
#             list1.clear()
#             # for index in range(len(list1)):
#     #             #     if list1[index]=='Null':
#     #             #         continue
#     #             #     print(list1)
#
#     # class Node(object):
#     #     def __init__(self, data):
#     #         self.data = data
#     #         self.children = []
#     #
#     #     def add_child(self, obj):
#     #         self.children.append(obj)










    # f=open("file.txt","r")
    # str=[]
    # i=0
    #
    # contents=f.read()
    # str.append(contents)
    # print(str)
    # for i in contents:
    #     print(contents)
    #     i+=1
    # print(i)

    # fo = open("file.txt", "r+")
    # str = fo.readline()
    # str = str[1:5]
    # print("Read String is : ", str)
    # fo.close()
    # index=0
    # f=open("file.txt","r")
    # if f.mode=='r':
    #     contents=f.read()
    #     index+=1
    #     print(contents)

#     # for col in contents:
#     #     count=count+1
#     # print(count)
#
#     # while i<=5:
#     #     r=f.read()
#     #     print(r)
#     #     i=i+1
#     # else:
#     #     f.close()
#

#
#     for index in range(len(mystring2)):
#         print(index,mystring2[index])
#
#     index=0
#     while index<=1:
#         print(index,mystring2[index])
#         index+=1

if __name__ == "__main__":
    main()

    # n=Node()