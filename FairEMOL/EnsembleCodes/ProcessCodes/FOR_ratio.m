function [Res, detail] = FOR_ratio(Preds, idxs_sensitive, Truelabel, num_group)
    
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
        if sum((truelabel) .* (1-preds))==0
            res = [res 1];
        else
            if sum(1-preds) > 0
                res = [res sum((truelabel) .* (1-preds)) / sum(1-preds)];
            else
                if sum((truelabel) .* (1-preds))==0 &&  sum(1-preds)==0
                    res = [res 1];
                end
            end
        end
        detail = [detail, sum((truelabel) .* (1-preds)) , sum(1-preds)];
    end
    
    Res = get_ave_metrics(res, 2);
end

