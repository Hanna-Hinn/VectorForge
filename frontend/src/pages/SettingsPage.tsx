import { useQuery } from "@tanstack/react-query";
import { get } from "../api/client";
import type { SystemHealth, ProvidersInfo } from "../types/api";
import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";
import Spinner from "../components/ui/Spinner";
import Header from "../components/layout/Header";
import { t } from "../i18n";

export default function SettingsPage() {
  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ["status"],
    queryFn: () => get<SystemHealth>("/status"),
    refetchInterval: 15_000,
  });

  const { data: providers, isLoading: providersLoading } = useQuery({
    queryKey: ["providers"],
    queryFn: () => get<ProvidersInfo>("/status/providers"),
  });

  return (
    <>
      <Header title={t("settings.title")} />
      <div className="p-6 space-y-6 max-w-2xl">
        {/* Health */}
        <Card title={t("settings.health")}>
          {healthLoading ? (
            <Spinner />
          ) : health ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge
                  variant={
                    health.status === "healthy"
                      ? "success"
                      : health.status === "degraded"
                        ? "warning"
                        : "error"
                  }
                >
                  {health.status}
                </Badge>
              </div>
              {health.components.map((comp) => (
                <div
                  key={comp.name}
                  className="flex items-center justify-between rounded border border-gray-100 px-3 py-2 text-sm dark:border-gray-800"
                >
                  <span>{comp.name}</span>
                  <Badge
                    variant={comp.status === "healthy" ? "success" : "error"}
                  >
                    {comp.status}
                  </Badge>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-400">Unable to fetch health status.</p>
          )}
        </Card>

        {/* Providers */}
        <Card title={t("settings.providers")}>
          {providersLoading ? (
            <Spinner />
          ) : providers ? (
            <div className="space-y-3">
              <div>
                <h4 className="text-xs font-medium text-gray-500 mb-1">
                  Embedding Providers
                </h4>
                <div className="flex flex-wrap gap-1">
                  {providers.embedding_providers.map((p) => (
                    <Badge key={p}>{p}</Badge>
                  ))}
                </div>
              </div>
              <div>
                <h4 className="text-xs font-medium text-gray-500 mb-1">
                  LLM Providers
                </h4>
                <div className="flex flex-wrap gap-1">
                  {providers.llm_providers.map((p) => (
                    <Badge key={p}>{p}</Badge>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-gray-400">Unable to fetch provider info.</p>
          )}
        </Card>
      </div>
    </>
  );
}
