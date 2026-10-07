package com.personaengine.app.future;

import java.util.Map;
import java.util.UUID;

/**
 * Future extension contract for QLoRA fine-tuning and training job coordination.
 */
public interface TrainingService {

    UUID triggerTrainingJob(UUID personaId, Map<String, Object> hyperParameters);

    String getJobStatus(UUID jobId);
}
