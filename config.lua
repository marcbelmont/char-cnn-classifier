local Config = {}
local Data = require "data"

function Config.data(opt)
    local path = opt.cl.source
    local data, classes, idToChar = Data.init(opt, path, Data.convInputs)
    return Data.split(data, opt.all.trainingSplit), classes, idToChar
end

function Config.clOptions()
    local cmd = torch.CmdLine()
    cmd:option("-debug", 0)
    cmd:option("-restore", 0)
    cmd:option("-evaluate", 0)
    cmd:option("-source", "")
    local opt = {cl = cmd:parse(arg or {})}
    if opt.cl.evaluate == 1 and opt.cl.restore == 0 then
        print("Restore parameter must be set")
        os.exit()
    end
    return opt
end

function Config.init(opt)
    opt.all = {
        maxEpochs = 2,
        batchSize = 16,
        repeatData = 8,
        trainingSplit = .8
    }

    opt.seed = 1
    torch.manualSeed(opt.seed)

    opt.optimState = {
        decaySteps = 5 * 30000, -- Decay learning rate after N examples
        decay = .6,
        learningRate = .3
    }
    opt.optimMethod = optim.sgd
    opt.criterion = nn.ClassNLLCriterion()

    opt.data = {
        idToChar = {"~"},
        classes = {}
    }

    ----------------------
    -- Model parameters --
    ----------------------

    opt.convCuisine = {
        ingredient = {
            embedding = 15, -- embedding for each character
            stringSize = 300, -- shorter ingredients use padding
            convs = {{400, 5}}, -- # of recognized "words"
            dropout = 0.5,
        },
        hsize = {200},
    }

    opt.experimentInfo = string.format(
        "%s|%s|data=%s,split=%s",
        pprint.string(opt.convCuisine),
        pprint.string(opt.optimState),
        opt.all.repeatData,
        opt.all.trainingSplit
    ):gsub(" ", "")
    opt.bestCorrect = nil
    local size = 0
    for _, v in pairs(opt.convCuisine.ingredient.convs) do
        size = size + v[1]
    end
    opt.convCuisine.ingredient.size = size
    return opt
end

return Config
