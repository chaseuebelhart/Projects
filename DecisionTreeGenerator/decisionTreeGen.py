import math

#1's and 0's translate to True and False respectively

attribs = ['Rain', 'Res', 'Price', 'Pat', 'Alt', 'Bar', 'Hun', 'Type', 'Fri', 'Est']

examples = [
[0, 1, '$$$', 'Some', 1, 0, 1, 'French', 0, '0-10'],
 [0, 0, '$', 'Full', 1, 0, 1, 'Thai', 0, '30-60'],
 [0, 0, '$', 'Some', 0, 1, 0, 'Burger', 0, '0-10'],
 [0, 0, '$', 'Full', 1, 0, 1, 'Thai', 1, '10-30'],
 [0, 1, '$$$', 'Full', 1, 0, 0, 'French', 1, '>60'],
 [1, 1, '$$', 'Some', 0, 1, 1, 'Italian', 0, '0-10'],
 [1, 0, '$', 'None', 0, 1, 0, 'Burger', 0, '0-10'],
 [1, 1, '$$', 'Some', 0, 0, 1, 'Thai', 0, '0-10'],
 [1, 0, '$', 'Full', 0, 1, 0, 'Burger', 1, '>60'],
 [0, 1, '$$$', 'Full', 1, 1, 1, 'Italian', 1, '10-30'],
 [0, 0, '$', 'None', 0, 0, 0, 'Thai', 0, '0-10'],
 [0, 0, '$', 'Full', 1, 1, 1, 'Burger', 1, '30-60']]


#We use this form sometimes (transpose of the other list) so here's what it would look like after we zip
examples_by_attribs = [
[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
 [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
 ['$$$', '$', '$', '$', '$$$', '$$', '$', '$$', '$', '$$$', '$', '$'],
 ['Some', 'Full', 'Some', 'Full', 'Full', 'Some', 'None', 'Some', 'Full', 'Full', 'None', 'Full'],
 [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
 [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1],
 [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
 ['French', 'Thai', 'Burger', 'Thai', 'French', 'Italian', 'Burger', 'Thai', 'Burger', 'Italian', 'Thai', 'Burger'],
 [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1],
 ['0-10', '30-60', '0-10', '10-30', '>60', '0-10', '0-10', '0-10', '>60', '10-30', '0-10', '30-60']]  

start_target = [1,0,1,1,0,1,0,1,0,0,0,1]

def decision_tree_learning(exps, atts, par_exps, targs, par_targs):

    def count(item,itemSet):
        #Returns the number of occurences of item in the itemset
        n = 0
        for i in itemSet:
            if i == item:
                n+=1
        return n

    p = count(1,targs) #Num Positives
    n = count(0,targs) #Num Negatives  
                        

    def importance(at_lst, ex_lst):
        #Returns the value of the most important attribute
        mostImportant = 0
        at_name = 'null'

        def Gain(A):
            
            def B(q):
                if q == 0 or q == 1:
                    return(-math.log(1,2))
                else:
                    return -((q*(math.log(q,2)))+((1-q)*math.log(1-q,2)))
                

            def pknk(item, atr):
                nk = 0
                pk = 0
                for i in range(0,len(atr)):
                    if atr[i] == item:
                        if targs[i] == 0:
                            nk+=1
                        else:
                            pk+=1
                return (pk,nk)

            def remainder(A):
                #A is a list corresponding with that attribute
                sumR = 0
                setA = set(A)
                for d in setA:
                    pk, nk = pknk(d,A)
                    sumR += (((pk+nk)/(p+n))*B(pk/(nk+pk)))
                return sumR  
            
            return B(p/(p+n))-remainder(A)

        trans_list = list(map(list, zip(*ex_lst)))  #categorized by atribute not example
        i_best = 0
        for i in range(0,len(trans_list)):
            lst = trans_list[i]
            k = Gain(lst)
            if k > mostImportant:
                mostImportant = k
                i_best = i
                at_name = at_lst[i]
        return set(trans_list[i_best]),i_best, at_name #Return values of the most important

    def plurality_value(ex):
        pos = count(1,ex)
        neg = count(0,ex)
        if pos > neg:
            return 'True'
        else:
            return 'False'

    def same_class(ex):
        j = ex[0]
        for i in ex:
            if j != i:
                return 0
        return 1

    def unique_ex(ex, at, a_i, targs):
        ex_lst = []
        targ_lst = []
        for x in range(0,len(ex)):
            if ex[x][a_i] == at:
                ex_lst.append(ex[x])
                targ_lst.append(targs[x])
        return ex_lst, targ_lst       

    if not exps:
        return plurality_value(par_targs)
    elif same_class(targs):
        return plurality_value(targs)  #Targs will hold a list of the classification values relative to the examples
    elif not atts:
        return plurality_value(targs)
    else:
        a, a_index, a_name = importance(atts,exps)
        tree = {a_name:[]}
        del(atts[a_index])
        for vk in a:
            exs,newTargs = unique_ex(exps,vk, a_index, targs)
            nExs = list(map(list, zip(*exs)))
            del(nExs[a_index])
            nExs = list(map(list, zip(*nExs)))
            tree[a_name].append({vk:decision_tree_learning(nExs, atts, exps, newTargs, targs)})
        return tree

print(decision_tree_learning(examples, attribs, [], start_target, []))

            

    

    

    

    
            
