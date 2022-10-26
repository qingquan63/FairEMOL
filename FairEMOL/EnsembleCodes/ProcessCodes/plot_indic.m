function Metric_val = plot_indic(Pop, objs_idx, PF, flag)

    popobjs = Pop(:, objs_idx);
    gens = unique(Pop(:,1))';
    Metric_val = zeros(1,length(gens));
    gens_i = Pop(:,1);
    parfor gen_idx = 1:length(gens)
        gen = gens(gen_idx);
       pop = popobjs(gens_i == gen,:);
       if flag == 1
           Metric_val(gen_idx) = HV(pop, PF);
       else
           Metric_val(gen_idx) = IGD(pop, PF);
       end
        
    end


end