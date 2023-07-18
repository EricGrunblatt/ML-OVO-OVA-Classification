% Assignment 5: Multiclass Classification
% Eric Grunblatt

X = dlmread('X.txt'); %X: (1+d)*N; d=2 in this dataset for visualization; N=20 in this dataset
Y = dlmread('Y.txt'); %Y: 1*N
numRows = size(X,1);
numCols = size(X,2);

%%%%%% Part 1: One-Versus-All (OVA) %%%%%%
% Use logistic regression
alpha = 0.5;
epsilon = 0.0001;
ova_errors = 0;
figure;
title("One vs. All (OVA) Classification");
for a=1:max(Y(:))
    totalIterations = 0;
    w_logReg = zeros(1, numRows);
    while(totalIterations < 250)
        E_w = zeros(numRows, 1);
        totalIterations = totalIterations + 1;
        for i=1:numCols 
            % If Y(i) = class, then class = 1, o.w. -1
            if(Y(i) ~= a) 
                class = -1;
            else
                class = 1;
            end
            % Carry out function for one iteration
            x = exp(-class * w_logReg * X(:,i));
            sig = x/(x+1);
            E_w = E_w + (sig * (class * X(:,i)));
        end
        % Finish algorithm
        E_w = E_w / numCols;
        w_logReg = w_logReg + transpose(alpha * E_w);
        if((transpose(E_w) * E_w) < epsilon)
            break;
        end
    end
    % Creating/Plotting boundary line
    slope = -w_logReg(2)/w_logReg(3);
    intercept = -w_logReg(1)/w_logReg(3);
    %y =mx+c, m is slope and c is intercept(0)
    x = [min(X(2,:)),max(X(2,:))];
    y = (slope*x) + intercept;
    line(x, y, 'color', 'black');
    hold on
    for i=1:numCols
        if(Y(i) ~= a) 
            class = -1;
        else
            class = 1;
        end
        if(sign(w_logReg * X(:,i)) ~= sign(class))
            plot(X(2,i), X(3,i), 'x', 'color', 'red');
            ova_errors = ova_errors + 1;
        end
    end
end

% Plot the data
for i=1:numCols
    if(Y(i) == 1) % Y is shown as 1, blue circle
        plot(X(2,i), X(3,i), 'o', 'color', 'blue');
        hold on
    elseif(Y(i) == 2) % Y is shown as 2, green triangle
        plot(X(2,i), X(3,i), '^', 'color', 'green');
        hold on
    elseif(Y(i) == 3) % Y is shown as 3, magenta diamond 
        plot(X(2,i), X(3,i), 'd', 'color', 'magenta');
        hold on
    else % Y is shown as 4, cyan star 
        plot(X(2,i), X(3,i), 'p', 'color', 'cyan');
        hold on
    end
end
hold off

% Report error rate
ova_errors = ova_errors / (numCols * max(Y(:)));
fprintf('One vs. All (OVA) Error Rate: %f\n', ova_errors); 


%%%%%% Part 2: One-Versus-One (OVO) %%%%%%
array = 1:max(Y(:));
pairs = nchoosek(array, 2);
pairSize = size(pairs,1);
ovo_errors = 0;
weights_array = zeros(3,pairSize);
figure;
title("One vs. One (OVO) Classification");
% For all pairs with 1s
for a=1:pairSize
    totalIterations = 0;
    w_logReg = zeros(1, numRows);
    while(totalIterations < 150)
        E_w = zeros(numRows, 1);
        totalIterations = totalIterations + 1;
        for i=1:numCols 
            % If Y(i) = class, then class = 1, o.w. -1
            if(Y(i) == pairs(a,2))
                class = -1;
                % Carry out function for one iteration
                x = exp(-class * w_logReg * X(:,i));
                sig = x/(x+1);
                E_w = E_w + (sig * (class * X(:,i)));
            elseif(Y(i) == pairs(a,1))
                class = 1;
                % Carry out function for one iteration
                x = exp(-class * w_logReg * X(:,i));
                sig = x/(x+1);
                E_w = E_w + (sig * (class * X(:,i)));
            end
        end
        % Finish algorithm
        E_w = E_w / numCols;
        w_logReg = w_logReg + transpose(alpha * E_w);

        if((transpose(E_w) * E_w) < epsilon)
            break;
        end
    end
    % Creating/Plotting boundary line
    slope = -w_logReg(2)/w_logReg(3);
    intercept = -w_logReg(1)/w_logReg(3);
    %y =mx+c, m is slope and c is intercept(0)
    %x = [min(X(2,:)),max(X(2,:))];
    %y = (slope*x) + intercept;
    %line(x, y, 'color', 'black');
    hold on
    for i=1:numCols
        if(Y(i) == pairs(a,2)) 
            class = -1;
            if(sign(w_logReg * X(:,i)) ~= sign(class))
                plot(X(2,i), X(3,i), 'x', 'color', 'red');
                ovo_errors = ovo_errors + 1;
            end
        elseif(Y(i) == pairs(a,1))
            class = 1;
            if(sign(w_logReg * X(:,i)) ~= sign(class))
                plot(X(2,i), X(3,i), 'x', 'color', 'red');
                ovo_errors = ovo_errors + 1;
            end
        end      
    end
    weights_array(:,a) = transpose(w_logReg);
end

% Plot the data
for i=1:numCols
    if(Y(i) == 1) % Y is shown as 1, blue circle
        plot(X(2,i), X(3,i), 'o', 'color', 'blue');
        hold on
    elseif(Y(i) == 2) % Y is shown as 2, green triangle
        plot(X(2,i), X(3,i), '^', 'color', 'green');
        hold on
    elseif(Y(i) == 3) % Y is shown as 3, magenta diamond 
        plot(X(2,i), X(3,i), 'd', 'color', 'magenta');
        hold on
    else % Y is shown as 4, cyan star 
        plot(X(2,i), X(3,i), 'p', 'color', 'cyan');
        hold on
    end
end
ylim([min(X(3,:)) max(X(3,:))]);
hold off

% Report the error rate
fprintf('One vs. One (OVO) Error Rate: %f\n', ovo_errors/(numCols));

% Tournament Champions
champions = zeros(1,numCols);
for i=1:numCols
    new_votes = zeros(1,max(Y(:)));
    for j=1:pairSize
        if(sign(transpose(weights_array(:,j)) * X(:,i)) > 0)
            new_votes(pairs(j,2)) = new_votes(pairs(j,2)) + 1;
        else
            new_votes(pairs(j,1)) = new_votes(pairs(j,1)) + 1;
        end
    end
    %disp(new_votes);
    mostVotes = max(new_votes);
    index = 0;
    count = 0;
    for k=1:size(new_votes, 2)
        if(mostVotes == new_votes(k))
            index = k;
            count = count + 1;
        end
    end
    if(count > 1)
        champions(i) = 0;
    else
        champions(i) = index;
    end 
end

% Report the Tournament Champion
fprintf('One vs. One (OVO) Tournament Champion\n');
disp(champions);
