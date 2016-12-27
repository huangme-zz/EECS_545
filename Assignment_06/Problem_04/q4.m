function q4()
points = [0.1 0.15; 0.3 0.15; 0.12 0.17; 0.2 0.18; 0.3 0.17; 0.22 0.19; 0.45 0.2];
labels = [1;1;1;1;-1;-1;-1];
W = ones(7, 1) / 7;
err = sum(W .* (labels ~= h(points)));
alpha = 1/2 * log(())

function y = h(x)
    y = - x(:,2) + 0.17;

function p2()
points = [0.1 0.15; 0.3 0.15; 0.12 0.17; 0.2 0.18; 0.3 0.17; 0.22 0.19; 0.45 0.2];
labels = [1;1;1;1;-1;-1;-1];
positive_points = find(labels == 1);
negative_points = find(labels == -1);

M = 1;
n = size(labels, 1);
W = ones(n, 1) / n;
h = {};
alpha = [];
for t = 1:M
    err(t) = inf;
    thresholds = -2:0.01:2;
    for i = 1:length(thresholds)
        for j = -5:0.01:5
            for k = [1 2]
                tmph.dim = k;
                tmph.pos = j;
                tmph.threshold = thresholds(i);
                tmpe = sum(W.*(weakLearner(tmph, points) ~= labels));
                if (tmpe < err(t))
                    err(t) = tmpe;
                    h{t} = tmph;
                end
            end
        end
    end
    
    alpha(t) = 0.5 * log((1-err(t))/err(t));
    W = W .* exp(-alpha(t).*labels.*weakLearner(h{t}, points));
    W = W ./ sum(W);
end

function y = weakLearner(h, x)

if (h.pos > 1)
    y = double(x(:, h.dim) >= -h.threshold/h.pos);
else
    y = double(x(:, h.dim) < -h.threshold/h.pos);
end

pos = (y == 0);
y(pos) = -1;