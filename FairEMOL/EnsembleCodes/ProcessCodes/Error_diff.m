function [Res, detail] = Error_diff(Preds, idxs_sensitive, Truelabel, num_group)
    
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
        res(g) = sum(preds ~= truelabel) / length(preds);
        detail = [detail, sum(preds ~= truelabel) , length(preds)];
    end
    
    Res = get_ave_metrics(res, 1);
end

