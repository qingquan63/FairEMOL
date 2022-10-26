function PF = get_PF(pop, objs)

    
    FrontNo = NDSort(pop(:,objs), 1);
    PF = pop(FrontNo == 1,objs);
    


end