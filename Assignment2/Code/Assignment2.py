import numpy as np
import matplotlib.pyplot as plt

# transition probabilities
P_ij_A=np.array([[1/2,1/4,1/4],[1/16,3/4,3/16],[1/4,1/8,5/8]])
P_ij_B=np.array([[1/2,0,1/2],[1/16,7/8,1/16]])
P_ij_C=np.array([[1/4,1/4,1/2],[1/8,3/4,1/8],[3/4,1/16,3/16]])

# reward matrix
R_ij_A=np.array([[10,4,8],[8,2,4],[4,6,4]])
R_ij_B=np.array([[14,0,18],[8,16,8]])
R_ij_C=np.array([[10,2,8],[6,4,2],[4,0,8]])

N=20 #horizon length

def find_single_stage_cost(P_ij,g_ij):
    #has to be evaluated for each state
    E_cost=P_ij*g_ij #expected cost of at the current stage for the action
    # will get a matrix which has to be summed over the
    cost_summed=np.sum(E_cost,axis=1)
    return cost_summed #[cost(action1),cost(action2),cost(action3)]
    
J_k_arr=[] # to store the cost to go at each stage
J_N=[0,0,0] #The cost at the Nth stage.
J_k_arr.append(J_N) #array to store J in each stage

g_A=find_single_stage_cost(P_ij_A,R_ij_A)
g_B=find_single_stage_cost(P_ij_B,R_ij_B)
g_C=find_single_stage_cost(P_ij_C,R_ij_C)

actions_arr=[] #array to store actions optimal actions in each stage for each state
#actions_arr.append([np.argmax(g_A),np.argmax(g_B),np.argmax(g_C)])

for i in range(N):
    #print(J_k_arr[i])
    A_arr=g_A+[np.dot(J_k_arr[i],P_ij_A[0]),np.dot(J_k_arr[i],P_ij_A[1]),np.dot(J_k_arr[i],P_ij_A[2])]
    B_arr=g_B+[np.dot(J_k_arr[i],P_ij_B[0]),np.dot(J_k_arr[i],P_ij_B[1])]
    C_arr=g_C+[np.dot(J_k_arr[i],P_ij_C[0]),np.dot(J_k_arr[i],P_ij_C[1]),np.dot(J_k_arr[i],P_ij_C[2])]
    J_A=np.max(A_arr) #max function over the actions
    action_A=np.argmax(A_arr)
    J_B=np.max(B_arr) #max function over the actions
    action_B=np.argmax(B_arr)
    J_C=np.max(C_arr) #max function over the actions
    action_C=np.argmax(C_arr)
    J_k_arr.append([J_A,J_B,J_C])
    actions_arr.append([action_A,action_B,action_C])
    
print('The cost array (stage wise) is as follows')
print('*'*40)
print(J_k_arr[::-1])
print('*'*40)
print('The optimal actions (stage wise) are as follows')
print('*'*40)
print(actions_arr[::-1])
print('*'*40)

################################
## Gridworld

niter=40 # number of times the value iteration has to be performed
J=np.zeros((10,10)) # initialise J matrix with zeros for all states
JN=[] # contains the J matrix for each stage
JN.append(J)
goal=[9,3] # set the goal co-ordinate
up=np.zeros(np.shape(J))
left=np.zeros(np.shape(J))
down=np.zeros(np.shape(J))
right=np.zeros(np.shape(J))
actions_arr=[]
p=np.array([0.8,0.1,0.1]) # transition probability for [direction,direction + 90, direction - 90]
J_temp=np.zeros((np.shape(J)[0]+2,np.shape(J)[0]+2))
J[goal[0]][goal[1]]=100
for i in range(niter):
    J_temp[1:-1,1:-1]=np.copy(J)
    
    up=0.8*(-1+J_temp[0:-2,1:-1])+0.1*(-1+J_temp[1:-1,2:])+0.1*(-1+J_temp[1:-1,0:-2])# up right left
    right=0.8*(-1+J_temp[1:-1,2:])+0.1*(-1+J_temp[2:,1:-1])+0.1*(-1+J_temp[0:-2,1:-1])# right down up
    down=0.8*(-1+J_temp[2:,1:-1])+0.1*(-1+J_temp[1:-1,0:-2])+0.1*(-1+J_temp[1:-1,2:])# down left right
    left=0.8*(-1+J_temp[1:-1,0:-2])+0.1*(-1+J_temp[0:-2,1:-1])+0.1*(-1+J_temp[2:,1:-1])# left up down
    
    cost=np.dstack((up,right))
    cost=np.dstack((cost,down))
    cost=np.dstack((cost,left))
    J=np.max(cost,axis=2)
    actions=np.argmax(cost,axis=2)
    J[goal[0]][goal[1]]=100 #goal cost is always 100
    #worm holes
    J[3][2]=J[9][0]
    J[4][2]=J[9][0]
    J[5][2]=J[9][0]
    J[6][2]=J[9][0]
    J[8][7]=J[0][7]
    J_temp[:,0]=J_temp[:,1] #changing the left padding
    J_temp[:,-1]=J_temp[:,-2] #changing the right padding
    J_temp[0,1:-1]=J_temp[1,1:-1] #changing the upper padding
    J_temp[-1,1:-1]=J_temp[-2,1:-1] #changing the lower padding
    '''
    J_temp[0,:]=J_temp[1,:]
    J_temp[-1,:]=J_temp[-2,:]
    J_temp[1:-2,0]=J_temp[1:-2,1]
    J_temp[1:-2,-1]=J_temp[1:-2,-2]
    '''
    actions_arr.append(actions)
    JN.append(J) # to keep track of the evolution of J

max_arr=[]
for i in range(np.array(JN).shape[0]-1):
    J_i=JN[i]
    J_i_1=JN[i+1]
    max_arr.append(np.max(np.abs(J_i_1-J_i)))
    
plt.plot(max_arr)
plt.grid()
plt.title('Plot of $\max_a | J_{i+1}(s)-J_i(s)$| vs number of iterations',fontsize=13)
plt.xlabel('Number of iterations')
plt.ylabel('Difference $\delta_i$')
plt.show()

plt.rcParams['figure.figsize'] = [10, 10]
fig, ax = plt.subplots()
k=40
ax.matshow(J_temp)
for i in range(12):
    for j in range(12):
        c = int(J_temp[j,i])
        ax.text(i, j, str(c), va='center', ha='center')
#plt.imshow(J)
#plt.colorbar()
plt.show()