function preds = modelPredictions(dlnet,mbq)

    preds = [];    
    
    while hasdata(mbq)
        [dlX1,dlX2,~] = next(mbq);
        
        % Make predictions using the model function.
        YPred = predict(dlnet,dlX1,dlX2);
        

        preds = [preds YPred];
                
    end

end