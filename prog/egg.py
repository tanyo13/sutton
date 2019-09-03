# In[7]:


# Eggs

numFloors = 3
gtStates = [(egg, lower, upper) for egg in range(3)
                                for lower in range(1, numFloors+1)
                                for upper in range(lower+1, numFloors+1)]
gtActions = list(range(1, numFloors))

def func_dyn(s, a):
    (egg, lower, upper) = s
    if egg <= 0: 
        return None
    if a < lower:
        return (egg, lower, upper)
    if a >= upper:
        return (egg-1, lower, upper)
    p1 = (a - lower + 1) / (upper - lower)
    return [(p1, ((egg, lower, a), -1)), (1 - p1, ((egg - 1, a, upper), -1))]

mdp_eggs = MDP(gtStates, gtActions, func_dyn)


# In[ ]:




