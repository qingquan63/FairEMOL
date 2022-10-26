function non_pop =  get_nondominated(pop)

    FrontNo = NDSort(pop,1);
    non_pop = pop(FrontNo==1,:);


end