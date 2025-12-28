# Test knowledge acquisition logging end-to-end

Write-Host "=== Testing Knowledge Acquisition Logging ===" -ForegroundColor Cyan

# 1. Create world
Write-Host "`n1. Creating EarthWorld..." -ForegroundColor Yellow
$world = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/worlds" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"world_id":"earth-test-1","name":"Earth Test World","world_type":"earth","observation_dim":256,"action_dim":64}'
Write-Host "World created: $($world.id)" -ForegroundColor Green

# 2. Create agent
Write-Host "`n2. Creating agent..." -ForegroundColor Yellow
$agent = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/agents" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"name":"A1","config":{}}'
Write-Host "Agent created: $($agent.id) - $($agent.name)" -ForegroundColor Green

# 3. Assign agent to world
Write-Host "`n3. Assigning agent to world..." -ForegroundColor Yellow
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/agents/$($agent.id)/assign-world" -Method POST -Headers @{"Content-Type"="application/json"} -Body "{`"world_id`":`"$($world.id)`"}" | Out-Null
Write-Host "Agent assigned" -ForegroundColor Green

# 4. Start simulation
Write-Host "`n4. Starting simulation..." -ForegroundColor Yellow
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/simulation/start" -Method POST | Out-Null
Write-Host "Simulation started" -ForegroundColor Green

# 5. Wait for knowledge acquisition
Write-Host "`n5. Waiting 30 seconds for agent to explore..." -ForegroundColor Yellow
for ($i = 30; $i -gt 0; $i--) {
    Write-Host "`rTime remaining: $i seconds " -NoNewline
    Start-Sleep -Seconds 1
}
Write-Host ""

# 6. Check simulation status
Write-Host "`n6. Checking simulation status..." -ForegroundColor Yellow
$status = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/simulation/status"
Write-Host "Status: $($status.status)" -ForegroundColor Cyan
Write-Host "Total steps: $($status.stats.total_steps)" -ForegroundColor Cyan
Write-Host "Agents: $($status.stats.num_agents)" -ForegroundColor Cyan

# 7. Check agent metrics
Write-Host "`n7. Checking agent metrics..." -ForegroundColor Yellow
$agent_state = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/agents/$($agent.id)"
Write-Host "Steps: $($agent_state.metrics.total_steps)" -ForegroundColor Cyan
Write-Host "Topics explored: $($agent_state.metrics.topics_explored)" -ForegroundColor Cyan
Write-Host "Knowledge sources queried: $($agent_state.metrics.knowledge_sources_queried)" -ForegroundColor Cyan
Write-Host "Knowledge acquired: $($agent_state.metrics.knowledge_acquired)" -ForegroundColor Cyan

# 8. Check knowledge_acquisition_log database table
Write-Host "`n8. Checking knowledge_acquisition_log..." -ForegroundColor Yellow
$log_count = docker compose exec -T db psql -U earthlink -d earthlink -t -c "SELECT COUNT(*) FROM knowledge_acquisition_log;"
Write-Host "Database log entries: $($log_count.Trim())" -ForegroundColor Cyan

if ($log_count.Trim() -gt 0) {
    Write-Host "`n=== DATABASE ENTRIES FOUND! ===" -ForegroundColor Green
    Write-Host "`nSample entries:" -ForegroundColor Yellow
    docker compose exec -T db psql -U earthlink -d earthlink -c "SELECT agent_id, topic, source, knowledge_count, created_at FROM knowledge_acquisition_log ORDER BY created_at DESC LIMIT 5;"
} else {
    Write-Host "`n=== NO DATABASE ENTRIES (PROBLEM) ===" -ForegroundColor Red
    Write-Host "`nChecking logs for errors..." -ForegroundColor Yellow
    docker compose logs api --tail 50 | Select-String -Pattern "Error in agent|Failed to log|RuntimeError|knowledge" -Context 2
}

# 9. Stop simulation
Write-Host "`n9. Stopping simulation..." -ForegroundColor Yellow
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/simulation/stop" -Method POST | Out-Null
Write-Host "Simulation stopped" -ForegroundColor Green

Write-Host "`n=== Test Complete ===" -ForegroundColor Cyan
