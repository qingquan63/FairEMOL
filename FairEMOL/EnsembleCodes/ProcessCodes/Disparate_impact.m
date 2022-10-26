function [Res, detail] = Disparate_impact(Preds, idxs_sensitive, Truelabel, num_group)
    
    Preds = Preds(:);
    Truelabel = double(Truelabel(:));
    Preds(Preds >= 0.5) = 1;
    Preds(Preds < 0.5) = 0;
    res = [];
    detail = [];
    for g = num_group
        gr = idxs_sensitive == g;
        truelabel = Truelabel(gr);
        preds = Preds(gr);
        res(g) = sum(preds) / sum(gr);
        detail = [detail, sum(preds) , sum(gr)];
    end
    
    Res = get_ave_metrics(res, 2);
end

