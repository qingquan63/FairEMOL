function Res = Group_fairness(Preds, idxs_sensitive, Truelabel, num_group)
    
    alpha = 2;
    Preds = Preds(:);
    Truelabel = double(Truelabel(:));
    b = Preds - Truelabel + 1;
    Final_b = b;
    for g = num_group
        gr = idxs_sensitive == g;
        Final_b(gr) = mean(b(gr));

    end
    
    Res =  mean(power((Final_b / mean(Final_b)),alpha) - 1) / (alpha * (alpha - 1));
    
end

