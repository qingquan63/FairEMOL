clc,clear
cd(fileparts(mfilename('fullpath')));
addpath(genpath(cd));

%% metric info
optimized_metrics = {'Error','Average_odd_diff','Error_diff','Discovery_ratio','Predictive_equality','FOR_diff','FOR_ratio','FNR_diff','FNR_ratio'};
nonoptimized_metrics = {'Error_ratio','Discovery_diff','FPR_ratio','Disparate_impact','Statistical_parity','Equal_opportunity','Equalized_odds','Predictive_parity', 'Individual_fairness', 'Group_fairness','Accuracy'};

datanames ={'student','german','compas','LSAT','default','adult','bank','dutch','student_academics','heart_failure','diabetes','student_performance','IBM_employee', 'drug_consumption','patient_treatment'};
% 1:  best
% 2:  knee
% 3:  second
% 4:  all the 
% 5:  diversity
% 6:  best + second  
%%
MOO_idx = [1 2 3 4 5 6 7];
MOO_res_all = {};
SOO_res_all = {};
labels = {'EnsBest','EnsAll','EnsKnee','EnsDiv','KCR','KCS','LrKSCR','LrKLSCR','KCSRN'};
%%
for data_idx = 1:length(datanames)
    dataname = datanames{data_idx}
    run_paths = get_filenames([dataname,'_info']);
    
    Ensemble_mat = [];
    for run_idx = 1:30
        temp_name = run_paths{run_idx};
        data_exp = load(temp_name(1:(end-1))); Info = data_exp.Info;
        Ensemble_mat = [Ensemble_mat;Info.Metric_ensemble_value_test];
    end
    
    moo_res_all = {};
    for pla_idx = 1:length(MOO_idx)
        plan_idx = MOO_idx(pla_idx);
        MOO = Ensemble_mat(Ensemble_mat(:,1) == plan_idx,[21 3:18]);
        moo_res_all = [moo_res_all MOO];
    end
   MOO_res_all = [MOO_res_all;moo_res_all];
   
    Other_ensemble = [];
    counter = 0;
    for soft_hard = 0
        for idx = [0 1 2 3 4]
           other_dataname = [dataname, '/', 'metrics_objs_test_softhard%d_algo%d.txt'];
           res = load(sprintf(other_dataname, soft_hard, idx));
           Other_ensemble = [Other_ensemble; [ones(size(res,1),1)*counter,res]];  
           counter = counter + 1;
        end
    end
   counter = counter - 1;
  
   soo_res_all = {};
    for pla_idx = 0:counter
        flag = Other_ensemble(:,1) == pla_idx;
        flag_temp = cumsum(flag);
        ed = find(flag_temp == 51);
        flag(ed:end) = 0;
        SOO = Other_ensemble(flag,[19, 3:18]);
        soo_res_all = [ soo_res_all SOO];
    end
   SOO_res_all = [SOO_res_all;soo_res_all];
   
end   





%%
function files_names = get_filenames(pathname)

    files = dir(fullfile(pathname));
    files(1:2) = [];
    files_names = {};
    for i = 1:length(files)
        files_names{i} = [pathname, '/', files(i).name '/'];
    end

end

function Latex_code = Latex_transfer(mean_value, std_value, h_values, color_type)
    
    % \cellcolor[rgb]{ .851+  .851+  .851}
    num = length(mean_value);
    Latex_code = '';
    for i = 1:num
        if isempty(color_type)
            if (h_values(i) == -1)
                Latex_code = [Latex_code, num2str(mean_value(i),'%.5f'),' (', num2str(std_value(i),'%.3e'), ')-,'];
            elseif(h_values(i) == 0)
                Latex_code = [Latex_code, num2str(mean_value(i),'%.5f'),' (', num2str(std_value(i),'%.3e'), ')$\approx$,'];
            else
                Latex_code = [Latex_code, num2str(mean_value(i),'%.5f'),' (', num2str(std_value(i),'%.3e'), ')+,'];
            end
        else
            if any(ismember(i, color_type))
                if (h_values(i) == -1)
                    Latex_code = [Latex_code, '\cellcolor[rgb]{ .851+  .851+  .851}', num2str(mean_value(i),'%.5f'),' (', num2str(std_value(i),'%.3e'), ')-,'];
                elseif(h_values(i) == 0)
                    Latex_code = [Latex_code, '\cellcolor[rgb]{ .851+  .851+  .851}', num2str(mean_value(i),'%.5f'),' (', num2str(std_value(i),'%.3e'), ')$\approx$,'];
                else
                    Latex_code = [Latex_code, '\cellcolor[rgb]{ .851+  .851+  .851}', num2str(mean_value(i),'%.5f'),' (', num2str(std_value(i),'%.3e'), ')+,'];
                end
            else
                if (h_values(i) == -1)
                    Latex_code = [Latex_code, num2str(mean_value(i),'%.5f'),' (', num2str(std_value(i),'%.3e'), ')-,'];
                elseif(h_values(i) == 0)
                    Latex_code = [Latex_code, num2str(mean_value(i),'%.5f'),' (', num2str(std_value(i),'%.3e'), ')$\approx$,'];
                else
                    Latex_code = [Latex_code, num2str(mean_value(i),'%.5f'),' (', num2str(std_value(i),'%.3e'), ')+,'];
                end
            end
        end
    end
%     Latex_code = [Latex_code, '/n, '];


end

function Latex_code = Latex_transfer2(mean_value, std_value, color_type)

    num = length(mean_value);
    Latex_code = '';
    for i = 1:num
        if isempty(color_type)
            Latex_code = [Latex_code, num2str(mean_value(i),'%.5f'),' (', num2str(std_value(i),'%.3e'), '),'];
        else
            if any(ismember(i, color_type))
                Latex_code = [Latex_code, '\cellcolor[rgb]{ .851+  .851+  .851}', num2str(mean_value(i),'%.5f'),' (', num2str(std_value(i),'%.3e'), '),'];
            else
                Latex_code = [Latex_code, num2str(mean_value(i),'%.5f'),' (', num2str(std_value(i),'%.3e'), '),'];
            end
        end
    end
%     Latex_code = [Latex_code, '/n, '];


end

function Res = statis_cmp(alog1, alog2)
    
    N = size(alog1,2);
    Res = zeros(1,3);
    for i = 1:N
        [p,h] = ranksum(alog1(:,i),alog2(:,i));
        if h == 0
           Res(2) =  Res(2) + 1;
        else
           if mean(alog1(:,i)) < mean(alog2(:,i))
               Res(1) =  Res(1) + 1;
           elseif mean(alog1(:,i)) > mean(alog2(:,i))
               Res(3) =  Res(3) + 1;
           else
               Res(2) =  Res(2) + 1;
           end
        end
    end
    
end