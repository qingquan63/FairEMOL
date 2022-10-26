function pop = get_pop_gen(base_name, obj_idx, total_gen, gen, train_valid_test)
    

    switch train_valid_test
        case 1 
            file_names = '/allobjs/ALL_Objs_train_gen%d_sofar.csv';
        case 2
            file_names = '/allobjs/ALL_Objs_valid_gen%d_sofar.csv';
        case 3
            file_names = '/allobjs/ALL_Objs_test_gen%d_sofar.csv';
    end
    
    file_names = [base_name file_names];
    file_name = sprintf(file_names,total_gen);
    data = load(file_name);
    allpop = data(:,obj_idx);
    pop = allpop(data(:,1)==gen,:);

end
