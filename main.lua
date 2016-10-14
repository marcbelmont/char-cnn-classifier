require "torch"
require "pprint"
require "nngraph"
require "optim"

local Models = require "models"
local train = require "train"
local Config = require "config"
local Data = require "data"
local test = require "test"

local CHECKPOINT_PATH = "checkpoint-%s.t7"

-------------
-- Options --
-------------

local opt = Config.clOptions()
local data

if opt.cl.restore > 0 then
    -- Options
    local path = string.format(CHECKPOINT_PATH, opt.cl.restore)
    print("Loading model " .. path)
    local x = torch.load(path)
    x.cl = opt.cl
    opt = x
    torch.manualSeed(opt.seed)
    if opt.cl.evaluate == 0 then
        data, opt.data.classes, opt.data.idToChar = Config.data(opt)
    end
else
    -- Options
    opt = Config.init(opt)
    data, opt.data.classes, opt.data.idToChar = Config.data(opt)

    -- Model
    opt.model = Models.ConvCuisine(opt.convCuisine, opt.data)
    opt.params, opt.gradParams = opt.model:getParameters()
end

-----------
-- Train --
-----------

local function trainAndTest()
    local experimentId
    print(opt.experimentInfo)
    if opt.cl.restore == 0 then
        experimentId = 1
    end

    local bestLoss, bestEpoch
    for epoch = 1, opt.all.maxEpochs do
        print("Epoch " .. epoch)

        -- Train and test
        local validationLoss, validationCorrect = train(data.training, opt, data.validation)
        if not bestLoss or validationLoss < bestLoss then
            bestLoss = validationLoss
            bestEpoch = epoch
        end

        -- save best model
        if ((not opt.bestCorrect or
             validationCorrect > opt.bestCorrect)) then
            print("Saving...")
            opt.bestCorrect = validationCorrect
            torch.save(string.format(CHECKPOINT_PATH, experimentId), opt)
        end

        -- early stopping
        if epoch > 2 * bestEpoch and epoch > 10 then
            print("Early stopping!")
            break
        end
    end
end

local function evaluate()
    opt.all.repeatData = 1
    local data, classes = Data.init(opt, opt.cl.source, Data.convInputs)
    local validationConfusion = optim.ConfusionMatrix(classes)
    print(opt.experimentInfo)
    local validationCorrect, validationLoss = test(data, opt, 1)
    print("Correct", validationCorrect, "Loss", validationLoss)
end

if opt.cl.evaluate > 0 then
    evaluate()
else
    trainAndTest()
end
