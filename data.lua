require "pprint"
local cjson = require "cjson"

local Data = {}

----------------
-- Split data --
----------------

function Data.split(data, trainingSplit)
    local validationIndex = math.floor(trainingSplit * data.targets:size(1))
    data.training.inputs = data.inputs[{{1, validationIndex}}]
    data.validation.inputs = data.inputs[{{validationIndex + 1, data.targets:size(1)}}]
    data.training.targets = data.targets[{{1, validationIndex}}]
    data.validation.targets = data.targets[{{validationIndex + 1, data.targets:size(1)}}]
    return data
end

-----------------
-- Read inputs --
-----------------

function Data.convInputs(data, recipes, opt, shuffle)
    local len = #recipes * opt.all.repeatData
    local inputs = torch.Tensor(len, opt.convCuisine.ingredient.stringSize):fill(1)
    local ingredients = {}
    local idToChar = opt.data.idToChar
    local charToId = {}
    for k, v in pairs(idToChar)  do
        charToId[v] = k
    end
    local ingredientTooLong = 0

    -- Read recipes
    for i = 1, #recipes do
        -- Add all recipes and shuffle ingredients each time
        for k = 1, opt.all.repeatData do
            -- Put all ingredients in one string
            local ingredientString = ""
            local ingredientSize = #recipes[i].ingredients
            local shuffle2 = torch.randperm(ingredientSize)
            for j = 1, ingredientSize do
                local ingredient =  recipes[i].ingredients[shuffle2[j]]:lower():gsub("[^a-z ]", " ")
                ingredientString = ingredientString .. "|" .. ingredient
            end

            -- Encode each character
            local size = math.min(opt.convCuisine.ingredient.stringSize, #ingredientString)
            for j = 1, size do
                local char = ingredientString:sub(j, j)
                if charToId[char] == nil then
                    idToChar[#idToChar + 1] = char
                    charToId[char] = #idToChar
                end
                inputs[shuffle[i] * opt.all.repeatData - (k - 1)][j] = charToId[char]
            end

            -- Debug
            if #ingredientString > opt.convCuisine.ingredient.stringSize then
                ingredientTooLong = ingredientTooLong + 1
            end
            if opt.cl.evaluate == 1 then
                opt.debugString = ingredientString
                break
            end
        end
    end
    print(" Ingredient too long:", ingredientTooLong)
    return inputs, idToChar
end

------------------
-- Data loaders --
------------------

function Data.init(opt, path, inputReader)
    print("Loading data " .. path)

    local f = io.open(path, "r")
    local recipes = cjson.decode(f:read("*all"))
    local data = {
        training = {inputs = {}, targets = {}},
        validation = {inputs = {}, targets = {}}}

    local shuffle
    if opt.cl.debug == 1 then
        shuffle = torch.range(1, #recipes)
    else
        shuffle = torch.randperm(#recipes)
    end

    -- Load targets
    local targets = torch.Tensor(shuffle:size(1) * opt.all.repeatData)
    local ids = torch.Tensor(shuffle:size(1) * opt.all.repeatData)
    local classes = opt.data.classes
    local cuisines = {}
    for k, v in pairs(classes)  do
        cuisines[v] = k
    end
    local cuisineCount = #classes + 1
    local cuisineStats = {}
    for i = 1, #recipes do
        local key = recipes[i].cuisine
        for j = 1, opt.all.repeatData  do
            local index = shuffle[i] * opt.all.repeatData - (j - 1)
            if key == nil then
                targets[index] = 1
            else
                if cuisines[key] == nil then
                    cuisines[key] = cuisineCount
                    cuisineCount = cuisineCount + 1
                    table.insert(classes, key)
                end
                targets[index] = cuisines[key]
            end
            ids[index] = recipes[i].id
        end
        if key ~= nil then
            if cuisineStats[key] == nil then
                cuisineStats[key] = 0
            end
            cuisineStats[key] = cuisineStats[key] + 1
        end
    end
    pprint(cuisineStats)
    data.targets = targets
    data.ids = ids
    local inputs, idToChar = inputReader(data, recipes, opt, shuffle)
    data.inputs = inputs
    return data, classes, idToChar
end

return Data
