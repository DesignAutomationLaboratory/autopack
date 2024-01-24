local module = {}

local base64 = require("base64")
local inspect = require("inspect")
local msgpack = require("MessagePack")
-- IPS seems to use SLB (https://code.google.com/archive/p/slb/)
local slb = require("SLB")

local NaN = 0 / 0

local function _pack(data)
  return base64.encode(msgpack.pack(data))
end

local function _unpack(string)
  return msgpack.unpack(base64.decode(string))
end

local function runAndPack(runFunc)
  local function packFunc(success, result)
    return _pack({
      success = success,
      result = result,
    })
  end

  local function errHandler(err)
    return {
      error = err,
      traceback = debug.traceback(err, 2),
    }
  end

  local runSuccess, runResult = xpcall(runFunc, errHandler)
  if not runSuccess and type(runResult) ~= "table" then
    -- For some errors, like C++ exceptions, the error handler is not
    -- called
    runResult = errHandler(runResult)
  end
  local packSuccess, packResult = xpcall(packFunc, errHandler, runSuccess, runResult)

  if packSuccess then
    return packResult
  else
    return packFunc(packSuccess, packResult)
  end
end

local function _type(obj)
  -- Returns the type of `obj` as a string.
  -- Unlike `type`, this function also works for IPS userdata.
  local luaType = type(obj)
  if luaType == "userdata" then
    -- IPS objects are userdata without a metatable. Try to look up
    -- using SLB but fall back to "userdata"
    return slb.type(obj) or "userdata"
  else
    return luaType
  end
end

local function log(msg)
  -- Logs a message to the IPS log
  print("Autopack: " .. msg)
end

local function pause(msg)
  -- Pauses the script until the user presses enter.
  local answer = Ips.question((msg or "") .. "\n\nContinue?")
  if answer == false then
    error("Script aborted")
  end
end

local function vectorToTable(vector)
  local table = {}
  local vecType = _type(vector)
  local size
  if vecType == "Vector3d" or vecType == "Vector3i" then
    -- Vector3[id] does not have a size() method
    size = 3
  else
    size = vector:size()
  end

  for i = 1, size do
    -- IPS vectors are 0-indexed
    local element = vector[i - 1]

    -- If the element is a userdata, assume it's a vector for now
    if type(element) == "userdata" then
      table[i] = vectorToTable(element)
    else
      table[i] = element
    end
  end

  return table
end

local function range(from, to)
  -- Returns an array with values from `from` to `to`, inclusive.
  local arr = {}
  for v = from, to do
    arr[#arr + 1] = v
  end
  return arr
end

local function treeObjChildren(treeObj)
  local children = {}
  local numChildren = treeObj:getNumChildren()
  for i = 1, numChildren do
    if i == 1 then
      children[i] = treeObj:getFirstChild()
    else
      children[i] = children[i - 1]:getNextSibling()
    end
  end
  return children
end

local function loadAndFitScene(scenePath)
  log("Loading scene " .. scenePath)
  local loaded = Ips.loadScene(scenePath)
  -- Fitting the scene helps with two things:
  -- 1. The scene is loaded in the background, and this makes sure it's
  --    done before we continue
  -- 2. It makes it easier to see what's going on
  Ips.fitScene()
  return loaded
end

local function saveScene(scenePath)
  log("Saving scene " .. scenePath)
  return Ips.saveScene(scenePath)
end

local function clearScene()
  -- Clears the scene of all active objects, static geometry, measures,
  -- and mechanisms
  local roots = {
    -- Start with processes, as they may have dependencies that are
    -- active objects
    Ips.getProcessRoot(),
    Ips.getActiveObjectsRoot(),
    Ips.getGeometryRoot(),
    Ips.getMeasuresRoot(),
    Ips.getMechanismRoot(),
    Ips.getSimulationsRoot(),
  }
  for _, root in pairs(roots) do
    while root:getNumChildren() > 0 do
      local child = root:getLastChild()
      Ips.deleteTreeObject(child)
    end
  end
  log("Scene cleared")
end

local function getOrCreateActiveGroup(groupName, parent)
  parent = parent or Ips.getActiveObjectsRoot()
  local activeGroup = parent:findFirstExactMatch(groupName)
  if not activeGroup then
    activeGroup = Ips.createAssembly(groupName)
    Ips.moveTreeObject(activeGroup, parent)
  end
  return activeGroup
end

local function getOrCreateGeometryGroup(groupName, parent)
  parent = parent or Ips.getGeometryRoot()
  local geoGroup = parent:findFirstExactMatch(groupName)
  if not geoGroup then
    geoGroup = Ips.createGeometryGroup(parent)
    geoGroup:setLabel(groupName)
  end
  return geoGroup
end

local function createHarnessRouter(harnessSetup)
  local cableSim = CableSimulation()
  local harnessRouter = HarnessRouter()
  local treeObject = Ips.getActiveObjectsRoot()

  for _, cable in pairs(harnessSetup.cables) do
    local startNode = treeObject:findFirstMatch(cable.start_node)
    assert(startNode, "Could not find start node " .. cable.start_node)
    local startFrame = startNode:getFirstChild()
    local startVis = startFrame:toCableMountFrameVisualization()
    local endNode = treeObject:findFirstMatch(cable.end_node)
    assert(endNode, "Could not find end node " .. cable.end_node)
    local endFrame = endNode:getFirstChild()
    local endVis = endFrame:toCableMountFrameVisualization()
    local myCableType = cableSim:getComponentTemplate(cable.cable_type)
    harnessRouter:addSegmentTerminalMountFrames(startVis, endVis, myCableType)
  end

  for _, geometry in pairs(harnessSetup.geometries) do
    local envGeom
    if geometry.name == "Static Geometry" then
      envGeom = Ips.getGeometryRoot()
    else
      envGeom = treeObject:findFirstMatch(geometry.name)
      assert(envGeom, "Could not find geometry " .. geometry.name)
    end
    local pref
    if geometry.preference == "Near" then
      pref = 0
    elseif geometry.preference == "Avoid" then
      pref = 1
    else
      pref = 2
    end
    harnessRouter:addEnvironmentGeometry(envGeom, geometry.clearance / 1000, pref, geometry.clipable)
  end

  harnessRouter:setMinMaxClipClipDist(harnessSetup.clip_clip_dist[1], harnessSetup.clip_clip_dist[2])
  harnessRouter:setMinMaxBranchClipDist(harnessSetup.branch_clip_dist[1], harnessSetup.branch_clip_dist[2])
  harnessRouter:setEnableClipSettings(true)
  harnessRouter:setMinBoundingBox(harnessSetup.min_bounding_box)
  if harnessSetup.custom_bounding_box then
    local bBox = harnessSetup.custom_bounding_box
    harnessRouter:setBoundingBox(
      Vector3d(unpack(bBox[1])),
      Vector3d(unpack(bBox[2]))
    )
  end
  harnessRouter:computeGridSize(harnessSetup.grid_resolution)
  -- Must be set before building the cost field
  harnessRouter:setAllowInfeasibleTopologySolutions(harnessSetup.allow_infeasible_topology)
  harnessRouter:buildCostField()

  return harnessRouter
end

local function getCostField(harnessSetup)
  local harnessRouter = createHarnessRouter(harnessSetup)
  local gridSize = harnessRouter:getGridSize()
  local coords = {}
  local costs = {}
  -- Using 1-based indexing to get arrays when packing with msgpack
  for i_x = 1, gridSize[0], 1 do
    coords[i_x] = {}
    costs[i_x] = {}
    for i_y = 1, gridSize[1], 1 do
      coords[i_x][i_y] = {}
      costs[i_x][i_y] = {}
      for i_z = 1, gridSize[2], 1 do
        -- IPS uses 0-based indexing for grid nodes
        local coord = harnessRouter:getNodePosition(i_x - 1, i_y - 1, i_z - 1)
        local cost = harnessRouter:getNodeCost(i_x - 1, i_y - 1, i_z - 1)
        coords[i_x][i_y][i_z] = { coord.x, coord.y, coord.z }
        costs[i_x][i_y][i_z] = cost
      end
    end
  end
  return { coords = coords, costs = costs }
end

local function setHarnessRouterNodeCosts(harnessRouter, costsArray)
  -- Sets the costs of the harness router's nodes to the values in `costsArray`.
  -- The array should be 1-indexed, but the router's nodes are 0-indexed.
  -- i.e., `costsArray[1][1][1]` will be the cost of the router's node (0, 0, 0).
  for x, xCosts in ipairs(costsArray) do
    for y, yCosts in ipairs(xCosts) do
      for z, cost in ipairs(yCosts) do
        harnessRouter:setNodeCost(x - 1, y - 1, z - 1, cost)
      end
    end
  end
end

local function routeHarnesses(
  harnessSetup,
  costs,
  bundlingWeight,
  namePrefix
)
  local harnessActiveGroup = getOrCreateActiveGroup("Autopack harnesses")
  local harnessGeoGroup = getOrCreateGeometryGroup("Autopack harnesses")
  local centerlinesGroup = getOrCreateGeometryGroup("Centerlines", harnessGeoGroup)
  centerlinesGroup:setExpanded(false)
  local collisionVizGroup = getOrCreateGeometryGroup("Collision visualization", harnessGeoGroup)
  local failedCableSimGroup = getOrCreateActiveGroup("Failed cable objects", harnessGeoGroup)
  failedCableSimGroup:setExpanded(false)
  local infeasibleTopologyGroup = getOrCreateGeometryGroup("Infeasible topology", harnessGeoGroup)
  -- This will effectively hide the previous harnesses and let the
  -- current ones appear. We must do some bit of hiding, because IPS
  -- tends to crash otherwise when the number of created harnesses
  -- increase. Some amount of feedback is nice, though.
  harnessActiveGroup:setGroupEnabled(false)
  harnessGeoGroup:setGroupEnabled(false)

  local numCables = #harnessSetup.cables
  local router = createHarnessRouter(harnessSetup)
  -- Use the default costs if none are given
  setHarnessRouterNodeCosts(router, costs)
  router:setObjectiveWeights(1, bundlingWeight, bundlingWeight)
  router:routeHarness()

  local numSolutions = router:getNumSolutions()
  if numSolutions == 0 then
    log("No solutions found for case " .. namePrefix)
    return {}
  end
  local topologyFeasible = router:getSolutionsAreTopologyFeasible()

  if topologyFeasible then
    router:smoothHarness()
  end

  local solutions = {}
  for solIdx = 0, numSolutions - 1 do
    local solutionName = namePrefix .. "." .. solIdx
    log("Capturing solution " .. solutionName)

    local segments = {}
    local numSegments = router:getNumBundleSegments(solIdx)
    for segIdx = 0, numSegments - 1 do
      local presmoothCoords
      local smoothCoords
      local clipPositions

      if topologyFeasible then
        presmoothCoords = vectorToTable(router:getPresmoothSegment(solIdx, segIdx, false))
        smoothCoords = vectorToTable(router:getSmoothSegment(solIdx, segIdx, false))
        clipPositions = vectorToTable(router:getClipPositions(solIdx, segIdx))
      end

      segments[segIdx + 1] = {
        radius = router:getSegmentRadius(solIdx, segIdx),
        cables = vectorToTable(router:getCablesInSegment(solIdx, segIdx)),
        discreteNodes = vectorToTable(router:getDiscreteSegment(solIdx, segIdx, false)),
        presmoothCoords = presmoothCoords or {},
        smoothCoords = smoothCoords or {},
        clipPositions = clipPositions or {},
      }
    end

    local cableSegmentOrder = {}
    for cableIdx = 0, numCables - 1 do
      cableSegmentOrder[cableIdx + 1] = vectorToTable(router:getCableSegmentOrder(solIdx, cableIdx))
    end

    local lengthTotal
    local lengthInCollision

    if topologyFeasible then
      lengthTotal = router:getSmoothHarnessLengthTotal(solIdx)
      lengthInCollision = router:getSmoothHarnessLengthCollision(solIdx)

      if lengthInCollision > 0 then
        -- Shows colliding parts in red
        local collisionVizTreeVector = router:buildSmoothSegments(solIdx, true)
        local collisionViz = collisionVizTreeVector[0]:getParent()
        collisionViz:setLabel(solutionName)
        Ips.moveTreeObject(collisionViz, collisionVizGroup)
      end

      local centerlinesTreeVector = router:buildSmoothSegments(solIdx, false)
      local centerlines = centerlinesTreeVector[0]:getParent()
      centerlines:setLabel(solutionName)
      -- Always hide the centerlines
      centerlines:setEnabled(false)
      Ips.moveTreeObject(centerlines, centerlinesGroup)

      local builtCableSimulation = router:buildSimulationObject(solIdx, true)
      if builtCableSimulation:hasExpired() then
        log("Failed to build simulation object for solution " .. solutionName)
        getOrCreateActiveGroup(solutionName .. " (failed)", harnessActiveGroup)
        -- Create a substitute for the cable simulation
        local cableSimSubstituteTreeVector = router:buildSmoothSegments(solIdx, true)
        local cableSimSubstitute = cableSimSubstituteTreeVector[0]:getParent()
        cableSimSubstitute:setLabel(solutionName)
        Ips.moveTreeObject(cableSimSubstitute, failedCableSimGroup)
      else
        builtCableSimulation:setLabel(solutionName)
        Ips.moveTreeObject(builtCableSimulation, harnessActiveGroup)
      end
    else
      -- Shows infeasible topology: yellow (too far away from clipable
      -- surface), red (collision or clearance-infeasible)
      local infeasibleTopologyVizTreeVector = router:buildDiscreteSegments(solIdx)
      local infeasibleTopologyViz = infeasibleTopologyVizTreeVector[0]:getParent()
      infeasibleTopologyViz:setLabel(solutionName)
      Ips.moveTreeObject(infeasibleTopologyViz, infeasibleTopologyGroup)
    end

    -- Gather the solution data
    -- Note that we index by 1 here for packing reasons
    solutions[solIdx + 1] = {
      name = solutionName,
      topologyFeasible = topologyFeasible,
      segments = segments,
      numBranchPoints = router:getNumBranchPoints(solIdx),
      cableSegmentOrder = cableSegmentOrder,
      objectiveWeightBundling = router:getObjectiveWeightBundling(solIdx),
      solutionObjectiveBundling = router:getSolutionObjectiveBundling(solIdx),
      solutionObjectiveLength = router:getSolutionObjectiveLength(solIdx),
      lengthTotal = lengthTotal or NaN,
      lengthInCollision = lengthInCollision or NaN,
    }
  end

  return solutions
end


local function copyRigidBodyGeometry(rigidBody, destTreeObj)
  rigidBody:setLocked(false)
  for _, child in pairs(treeObjChildren(rigidBody)) do
    if child:isPositionedTreeObject() and not child:isFrame() then
      Ips.copyTreeObject(child, destTreeObj)
    end
  end
  rigidBody:setLocked(true)
end

local function moveGripPoint(gripPoint, worldVector)
  -- local r = Rot3(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0))
  -- local transf = Transf3(r, worldVector)
  -- Does not work
  -- gripPoint:setTarget(transf)
  -- gripPoint:getVisualization():setTWorld(transf)
  -- Almost works
  -- gripPoint:getVisualization():setTControl(transf)
  -- Works

  -- The object frame points to the wrist, so to get to the actual grip
  -- point, we must account for the offset
  local placementVector = worldVector - gripPoint:getOffset()
  local gripPointViz = gripPoint:getVisualization()
  gripPointViz:transform(placementVector.x, placementVector.y, placementVector.z, 0, 0, 0)

  assert(gripPointViz:getTControl().t:distance(worldVector) < 1e-6, "Grip point not moved correctly")
end

local function copyToStaticGeometry(activeObjNames)
  -- Copies the rigid bodies with the given names to the static geometry
  -- root
  local activeObjRoot = Ips.getActiveObjectsRoot()
  local destTreeObj = getOrCreateGeometryGroup("Autopack copied geometry")
  for _, activeObjName in pairs(activeObjNames) do
    local rigidBody = activeObjRoot:findFirstExactMatch(activeObjName):toRigidBodyObject()
    copyRigidBodyGeometry(rigidBody, destTreeObj)
  end
  return destTreeObj
end

local function getAllManikinFamilies()
  local msc = ManikinSimulationController()
  -- Manikin family IDs are UUIDs, not related to names or indices
  local manikinFamilyIds = vectorToTable(msc:getManikinFamilyIDs())
  local manikinFamilyNames = {}
  for _, manikinFamilyId in pairs(manikinFamilyIds) do
    local family = msc:getManikinFamily(manikinFamilyId)
    local familyViz = family:getVisualization()
    manikinFamilyNames[#manikinFamilyNames + 1] = {
      id = manikinFamilyId,
      name = familyViz:getLabel(),
      manikinNames = vectorToTable(family:getManikinNames()),
    }
  end

  return manikinFamilyNames
end

local function evalErgo(
  geoNames,
  manikinFamilyId,
  coords,
  gripTol,
  ergoStandards,
  enableRbpp,
  updateScreen,
  keepGenObj
)
  local copiedGeoGroup = copyToStaticGeometry(geoNames)

  local msc = ManikinSimulationController()
  local family = msc:getManikinFamily(manikinFamilyId)
  local familyViz = family:getVisualization()
  family:enableCollisionAvoidance()

  local gripPoint = msc:createGripPoint()
  local gripPointViz = gripPoint:getVisualization()
  -- genObjsRoot:insert(0, gripPoint)
  gripPoint:setGripConfiguration("Tip Pinch")
  gripPoint:setSymmetricRotationTolerances(math.huge, math.huge, math.huge)
  gripPoint:setSymmetricTranslationTolerances(gripTol, gripTol, gripTol)

  local opSequence = OperationSequence()
  opSequence:setLabel("Autopack ergo evaluation")
  local familyActor = opSequence:addFamilyActor(familyViz)
  familyActor:setCurrentStateAsStart()
  -- Add a pause action so we get a time where the manikin is steadily
  -- in its start state
  local pauseAction = opSequence:createManikinWaitAction(familyActor, 1e-6)
  local graspAction = opSequence:createManikinGraspAction(familyActor, gripPointViz)
  if enableRbpp then
    graspAction:enableRigidBodyPathPlanning()
  else
    graspAction:disableRigidBodyPathPlanning()
  end
  -- Add a release action as it seems to help with resetting the
  -- manikins properly
  local releaseAction = opSequence:createManikinReleaseAction(familyActor, gripPointViz)
  releaseAction:maintainCurrentPosture()

  local outputTable = {
    ergoStandards = ergoStandards,
    ergoValues = {},
    errorMsgs = {},
  }
  local replay -- Declared here to enable resetting after the loop
  local pauseActionEndTime
  for coordIdx, coord in pairs(coords) do
    outputTable.ergoValues[coordIdx] = {}
    outputTable.errorMsgs[coordIdx] = {}

    moveGripPoint(gripPoint, Vector3d(coord[1], coord[2], coord[3]))

    for handIdx, handName in pairs({ "left", "right" }) do
      log("Ergo evaluation: coord " .. coordIdx .. "/" .. #coords .. ", " .. handName .. " hand")
      gripPoint:setHand(handIdx - 1)
      -- This shows up for the user in the UI
      gripPointViz:setLabel("Coord " .. coordIdx .. ", " .. handName .. " hand")

      replay = opSequence:executeSequence()
      outputTable.errorMsgs[coordIdx][handIdx] = replay:getReplayErrorMessage(graspAction)
      pauseActionEndTime = replay:getActionEndTime(pauseAction)
      local graspActionEndTime = replay:getActionEndTime(graspAction)

      if updateScreen then
        Ips.updateScreen()
      end

      outputTable.ergoValues[coordIdx][handIdx] = {}
      for ergoStandardIdx, ergoStandard in pairs(ergoStandards) do
        local ergoValues = vectorToTable(replay:computeErgonomicScore(ergoStandard, graspActionEndTime,
          graspActionEndTime))
        outputTable.ergoValues[coordIdx][handIdx][ergoStandardIdx] = ergoValues
      end
    end
  end
  -- Make sure that the last replay is rewinded, to set the manikin back
  -- to its start state. Rewinding to 0.0 does not properly reset the
  -- start state.
  replay:setTime(pauseActionEndTime)

  if not keepGenObj then
    Ips.deleteTreeObject(opSequence)
    Ips.deleteTreeObject(gripPointViz)
    Ips.deleteTreeObject(copiedGeoGroup)
  end

  return outputTable
end

local function createColoredPointCloud(points, treeParent, treeObjName, replaceExisting, enabled)
  -- `points` is an array of arrays, where each sub-array is a point, described by 6 numbers:
  -- x, y, z, r, g, b
  local staticGeoRoot = Ips.getGeometryRoot()
  if not treeParent then
    treeParent = staticGeoRoot
  elseif type(treeParent) == "string" then
    treeParent = getOrCreateGeometryGroup(treeParent, staticGeoRoot)
  end
  if replaceExisting then
    local existingTreeObj = treeParent:findFirstExactMatch(treeObjName)
    if existingTreeObj then
      Ips.deleteTreeObject(existingTreeObj)
    end
  end
  local builder = GeometryBuilder()
  for pointIdx, point in pairs(points) do
    builder:pushVertex(point[1], point[2], point[3])
    builder:pushColor(point[4], point[5], point[6])
  end

  builder:buildPoints()
  local treeObj = staticGeoRoot:getLastChild()
  Ips.moveTreeObject(treeObj, treeParent)
  treeObj:setLabel(treeObjName)
  treeObj:setEnabled(enabled)

  return treeObj
end

module.type = _type
module.pack = _pack
module.unpack = _unpack
module.runAndPack = runAndPack
module.vectorToTable = vectorToTable
module.range = range

module.loadAndFitScene = loadAndFitScene
module.saveScene = saveScene
module.clearScene = clearScene
module.getCostField = getCostField
module.setHarnessRouterNodeCosts = setHarnessRouterNodeCosts
module.routeHarnesses = routeHarnesses
module.getAllManikinFamilies = getAllManikinFamilies
module.evalErgo = evalErgo
module.createColoredPointCloud = createColoredPointCloud

module.base64 = base64
module.inspect = inspect
module.msgpack = msgpack

return module
