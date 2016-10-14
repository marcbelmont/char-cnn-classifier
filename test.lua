require "csvigo"

local function explore(model, opt, pred, target)
    -- Dense l
    print(model.modules[4], "Sorted activation units:")
    denseLayer = model.modules[4].output
    local values, indexes = torch.sort(denseLayer, 2, true)
    print(indexes)

    -- Examine filters
    local all = {}
    for j, _ in pairs(model.modules[2].modules) do
        local mod = model.modules[2].modules[j]
        if string.format("%s", mod) == "nn.TemporalConvolution" then
            local width = mod.weight:size(2) / opt.convCuisine.ingredient.embedding
            print(mod, width)
            local str = opt.debugString
            for i = 1, mod.output:size(2) do
                local map = mod.output[{{}, i}]
                local probabilities, indexes = torch.sort(map, 1, true)
                local match = str:sub(indexes[1], indexes[1] + width - 1)
                table.insert(all, {probabilities[1], string.format("%5s  %3s  [%s]", string.format("%3.2f", probabilities[1]), indexes[1], match)})
            end
        end
    end
    table.sort(all, function(a, b) return a[1] > b[1] end)
    for k, v in pairs(all) do
        print(v[2])
    end

    -- Prediction
    local probabilities, indexes = torch.sort(torch.exp(pred), 1, true)
    print(string.format("%.2f%% %5s",
                        probabilities[1] * 100,
                        indexes[1] == target),
          opt.data.classes[indexes[1]],
          indexes[1] == target and "" or opt.data.classes[target])
    return true
end

local function test(data, opt)
    local model, criterion = opt.model, opt.criterion
    local confusion = optim.ConfusionMatrix(opt.data.classes)

    model:evaluate()
    confusion:zero()
    local losses = {}
    local dataSize = data.targets:size(1)
    if opt.cl.evaluate == 1 then
        dataSize = 1
    end
    local results = {}
    local shuffle
    local size = dataSize
    if opt.cl.evaluate == 1 then
        size = math.min(dataSize, 7000)
        shuffle = torch.randperm(dataSize)
    else
        shuffle = torch.range(1, dataSize)
    end
    for t = 1, size do
        -- get new sample
        local target = data.targets[shuffle[t]]
        local input = data.inputs[shuffle[t]]:clone()

        -- test sample
        local pred = model:forward(input)

        -- Explore results
        if opt.cl.evaluate == 1 then
            explore(model, opt, pred, target)
        end
        losses[t] = criterion:forward(pred, target)
        confusion:add(pred, target)
        local _, index = torch.sort(torch.exp(pred), 1, true)
        if opt.cl.evaluate == 2 then
            results[t] = {data.ids[shuffle[t]], opt.data.classes[index[1]]}
        end
    end

    -- log
    confusion:__tostring__()
    if opt.cl.debug ~= 1 then
        print(confusion)
    end
    local loss = torch.Tensor(losses):mean()
    return confusion.totalValid, loss
end

return test
