function [Res, detail] = Statistical_parity(Preds, idxs_sensitive, Truelabel, num_group)
    
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
        res(g) = 1 - sum(preds) / sum(gr);
        detail = [detail, sum(1-preds) , sum(gr)];
    end
    
    Res = get_ave_metrics(res, 1);
end

