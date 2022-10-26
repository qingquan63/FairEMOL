function [Res_mat, Details] = calculate_metric(base_path, select_preds, train_valid_test_flag, optimized_metrics, nonoptimized_metrics, select_obj)
    
    Details = [];
    [idxs_sensitive, truelabel] = train_valid_test(base_path, train_valid_test_flag);
    all_metrics = [optimized_metrics nonoptimized_metrics];
    num_indicators = length(all_metrics);
    num_data = size(select_preds,1);
    idxs_sensitive = int16(idxs_sensitive)+1;
    truelabel = int16(truelabel');
    num_group = unique(idxs_sensitive); %#ok<*NASGU>
    group_nums = zeros(1, length(num_group));
    distri = [];
    for g = num_group
        group_nums(g) = sum(idxs_sensitive == g);
        temps_sens = idxs_sensitive == g;
        temps_true = truelabel(temps_sens);
        distri = [ distri ; [sum(1 - temps_true)  sum(temps_true)]];
    end
    
    Res_mat = zeros(size(select_preds,1), num_indicators);
    for metri_idx = 1:num_indicators
        details1 = [];
        for pop_idx = 1:num_data 
            preds = select_preds(pop_idx,:);
            if strcmp(all_metrics{metri_idx}, 'Error')
                temp = cross_entropy(preds,truelabel);
                
            elseif strcmp(all_metrics{metri_idx}, 'Equalized_odds')
                [temp,detail] = Equalized_odds(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'Equalized_odds')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'Error_diff')
                [temp,detail] = Error_diff(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'Error_diff')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'Discovery_ratio')
                [temp,detail] = Discovery_ratio(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'Discovery_ratio')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'Predictive_equality')
                [temp,detail] = Predictive_equality(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'Predictive_equality')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'FOR_ratio')
                [temp,detail] = FOR_ratio(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'FOR_ratio')
                    details1 = [details1; [pop_idx detail]];
                end

            elseif strcmp(all_metrics{metri_idx}, 'FNR_diff')
                [temp,detail] = FNR_diff(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'FNR_diff')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'FNR_ratio')
                [temp,detail] = FNR_ratio(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'FNR_ratio')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'Error_ratio')
                [temp,detail] = Error_ratio(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'Error_ratio')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'Discovery_diff')
                [temp,detail] = Discovery_diff(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'Discovery_diff')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'FPR_ratio')
                [temp,detail] = FPR_ratio(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'FPR_ratio')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'Disparate_impact')
                [temp,detail] = Disparate_impact(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'Disparate_impact')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'Statistical_parity')
                [temp,detail] = Statistical_parity(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'Statistical_parity')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'Equal_opportunity')
                [temp,detail]  = Equal_opportunity(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'Equal_opportunity')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'Average_odd_diff')
                [temp,detail] = Average_odd_diff(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'Average_odd_diff')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'Predictive_parity')
                [temp,detail] = Predictive_parity(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'Predictive_parity')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'FOR_diff')
                [temp,detail] = FOR_diff(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'FOR_diff')
                    details1 = [details1; [pop_idx detail]];
                end
            
            elseif strcmp(all_metrics{metri_idx}, 'Accuracy')
                temp = Accuracy(preds, idxs_sensitive, truelabel, num_group);
            
            elseif strcmp(all_metrics{metri_idx}, 'Individual_fairness')
                temp = Individual_fairness(preds, idxs_sensitive, truelabel, num_group);
                if  strcmp(select_obj, 'Individual_fairness')
                    details1 = [details1; [pop_idx detail]];
                end
                
            elseif strcmp(all_metrics{metri_idx}, 'Group_fairness')
                temp = Group_fairness(preds, idxs_sensitive, truelabel, num_group);
                
            end
            
            Res_mat(pop_idx,metri_idx) = temp;
            
        end  
        Details = [Details;details1];
    end

end