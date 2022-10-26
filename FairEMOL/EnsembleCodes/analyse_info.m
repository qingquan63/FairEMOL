clc,clear
load('MOO_SOO_res_all_test.mat')

datanames ={'student','german','compas','LSAT','default','adult','bank','dutch','student_academics','heart_failure','diabetes','student_performance','IBM_employee', 'drug_consumption','patient_treatment'};
labels = {'EnsBest','EnsAll','EnsKnee','EnsDiv','KCR','KCS','LrKSCR','LrKLSCR','KCSRN'};
% 1:  best (1)
% 2:  knee (3)
% 3:  second
% 4:  all the (2)
% 5:  diversity (4)
% 6:  best + second  
% 7:  best but delete repeat
select_ens_idx = [1 4 2 5];

All_average_ranks = [];

objs = [1 2 3 4 5 6 7 8 9];
All_mean_values = [];
All_std_values = [];
All_single_runs = [];
Ranks = [];
H_all = [];
DD = [];
for data_id = 1:length(datanames)
    dataname = datanames{data_id};
    moo_res = MOO_res_all(data_id,:);
    soo_res = SOO_res_all(data_id,:);
    MOO_single_res = [];
    SOO_single_res = [];
    for moo_idx = 1:size(select_ens_idx,2)
        moo_id = select_ens_idx(moo_idx);
        res = moo_res{moo_id};
        res = res(:,objs);
        res(:,1) = 1 - res(:,1);
        if(any(any(res == 0)))
            res = res + 0.00000000000000000000001;
        end
        moo_single_res = power(prod(res,2), 1/size(res,2));
        MOO_single_res = [MOO_single_res, moo_single_res];
    end
    
    for soo_id = 1:size(soo_res,2)
        res = soo_res{soo_id};
        res = res(:,objs);
        res(:,1) = 1 - res(:,1);
        if(any(any(res == 0)))
            res = res + 0.00000000000000000000001;
        end
        soo_single_res = power(prod(res,2), 1/size(res,2));
        SOO_single_res = [SOO_single_res, soo_single_res];
    end

        a = 1:30;

    All_mean_values = [All_mean_values;[mean(MOO_single_res(a,:),1),mean(SOO_single_res,1)]];
    All_std_values = [All_std_values;[std(MOO_single_res(a,:),1),std(SOO_single_res,1)]];
    dd = [MOO_single_res(a,:),SOO_single_res];
    H_all = [H_all;ranksum_mine(dd)];
    DD = [DD;dd];

    temp1 = mean([MOO_single_res(a,:),SOO_single_res],1);
    [~,temp_all] = sort(temp1);
    [~, temp_all] = sort(temp_all,2);

    All_average_ranks = [All_average_ranks;temp_all];
    All_single_runs = [All_single_runs;dd];
    
    moo_res = MOO_res_all(data_id,:);
    soo_res = SOO_res_all(data_id,:);
    all_mean_objs = [];
    for moo_i = 1:length(select_ens_idx)
        moo_allobjs = moo_res{select_ens_idx(moo_i)};
        moo_allobjs(:,1) = 1 - moo_allobjs(:,1);
        all_mean_objs = [all_mean_objs; mean(moo_allobjs,1)];
    end
    
    for soo_i = 1:5
        soo_allobjs = soo_res{soo_i};
        soo_allobjs(:,1) = 1 - soo_allobjs(:,1);
        all_mean_objs = [all_mean_objs; mean(soo_allobjs,1)];
    end
    
    [~,rank] = sort(all_mean_objs,1);
    [~,rank] = sort(rank,1);
    Ranks = [Ranks; [ones(size(rank,1),1) * data_id, rank]];
end

CD = criticaldifference(All_average_ranks,labels,0.05);
idx = repmat(1:size(All_average_ranks,2),size(All_average_ranks,1),1);
dunn(All_average_ranks(:)',idx(:)',1)

fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
% print('-painters','-dpdf','-r600','Nemenyi_new.pdf')

%%
res = zeros(size(All_average_ranks,2),17);
idxx = (unique(Ranks(:,1)))';
for i = idxx
    rank = Ranks(Ranks(:,1) == i, :);
    res = res + rank(:,2:end);
end
objs_means = res/length(unique(Ranks(:,1)));

Rankrank = [];
for i = 1:size(objs_means,2)
    [~,~,rankrank] = unique(objs_means(:,i));
    Rankrank = [Rankrank,rankrank];
end
figure
cmap=colormap('hot');

subplot(1,2,1)
heatmap(objs_means,'Colormap',cmap)
subplot(1,2,2)
heatmap(Rankrank,'Colormap',cmap)

% save('All_ranks_values.txt','objs_means','-ascii')
% save('All_ranks.txt','Rankrank','-ascii')

%% 计算p-values  每一个目标下
P_values = [];
for obj_idx = 1:17
    ranks = [];
    for data_id = 1:length(datanames)
        rank = Ranks(Ranks(:,1) == data_id, obj_idx+1);
        ranks = [ranks;rank'];
    end

    [p,tbl,stats] = friedman(ranks, 1,'off');
    c = multcompare(stats,'Alpha',0.05);
    fig = gcf;
    fig.PaperPositionMode = 'auto';
    fig_pos = fig.PaperPosition;
    fig.PaperSize = [fig_pos(3) fig_pos(4)];
%     print('-painters','-dpdf','-r600',['all_krus_pics/each', num2str(obj_idx), '.pdf'])
            
    P_values = [P_values p];
 end
[P_values;Rankrank]
%%

filename = 'Gmean_table.csv';
fid = fopen(filename, 'w');

Rank_of_mean_Gmeans = [];
for i = 1:size(All_mean_values,1)
    [~,~,rankrank] = unique(All_mean_values(i,:) );
    Rank_of_mean_Gmeans = [Rank_of_mean_Gmeans;rankrank'];
end

for algo_idx = 1:length(datanames)
    latex_res = Latex_transfer(All_mean_values(algo_idx,:), All_std_values(algo_idx,:), Rank_of_mean_Gmeans(algo_idx,:), H_all(algo_idx,:));
    fprintf(fid, ['%s\n'],['\emph{', datanames{algo_idx},'},',latex_res]);
end

fclose(fid);

%%


function Latex_code = Latex_transfer(mean_value, std_value, color_type, H_all)
    
    simply = {'','+','$\approx$','-'};
    num = length(mean_value);
    Latex_code = '';
    for i = 1:num
        if isempty(color_type)
            Latex_code = [Latex_code, num2str(mean_value(i),'%.5f'),'(', num2str(std_value(i),'%.3e'), simply{H_all(i)} , ')',','];
        else
            if color_type(i) == 1
                Latex_code = [Latex_code, '\cellcolor[rgb]{ .751+  .751+  .751+}', num2str(mean_value(i),'%.5f'),'(', num2str(std_value(i),'%.3e'), ')', simply{H_all(i)},','];
            elseif color_type(i) == 2
                Latex_code = [Latex_code, '\cellcolor[rgb]{ .880+  .880+  .880+}', num2str(mean_value(i),'%.5f'),'(', num2str(std_value(i),'%.3e'), ')', simply{H_all(i)},','];    
            else
                Latex_code = [Latex_code, num2str(mean_value(i),'%.5f'),'(', num2str(std_value(i),'%.3e'), ')', simply{H_all(i)},','];
            end
        end
    end
%     Latex_code = [Latex_code, '/n, '];


end



function H = ranksum_mine(metrics)
    % 1: none
    % 2: good
    % 3: similar
    % 4: worse
    n = size(metrics,2);
    H = [1];
    for i = 2:n
        [p, h] = ranksum(metrics(:,1), metrics(:,i));
        if h == 0
            H = [H 3];
        else
            if mean(metrics(:,1)) < mean(metrics(:,i))
                H = [H 4];
            else
                H = [H 2];
            end
        end
    end

end
