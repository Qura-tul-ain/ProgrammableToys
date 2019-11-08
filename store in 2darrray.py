# B[0][0]=S
# B[0][1]=O
# B[0][2]=B
# B[0][3]=O
# B[1][0]=O
# B[1][1]=B
# B[1][2]=O
# B[1][3]=B
# B[2][0]=O
# B[2][1]=O
# B[2][2]=O
# B[2][3]=G
# for item in B[i]:
#     for item1 in last:
# print('element at row index 1 & column index 2 is : ', num)
# B=[][]
# for item in B[i]:
#     for item1 in B[j]:
# Creates a list containing 5 lists, each of 8 items, all set to 0
w, h = 4, 3;# h=rows, w=columns
B = [[0 for x in range(w)] for y in range(h)]
B[0][0]='S'
B[0][1]='O'
B[0][2]='B'
B[0][3]='O'
B[1][0]='O'
B[1][1]='B'
B[1][2]='O'
B[1][3]='B'
B[2][0]='O'
B[2][1]='O'
B[2][2]='O'
B[2][3]='G'
print(B)


def NextState(Curr_State,Action):
    Curr_state = 0,0
    North = 360
    if Action==North:
        r,c==Curr_state
        r=r+1
        return(r,c)=='S'
    print(r,c)




if __name__ == "__NextState__":
        NextState(r,c,360)
