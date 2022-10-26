function Res = Individual_fairness(Preds, idxs_sensitive, Truelabel, num_group)
    
    alpha = 2;
    Preds = Preds(:);
    Truelabel = double(Truelabel(:));
    b = Preds - Truelabel + 1;
    
    Res =  mean(power((b / mean(b)),alpha) - 1) / (alpha * (alpha - 1));
    
end

