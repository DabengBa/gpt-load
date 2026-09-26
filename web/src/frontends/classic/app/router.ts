import type { Component } from 'vue'
import type { Router, RouterHistory, RouteRecordRaw, RouteRecordSingleView } from 'vue-router'
import { createRouter, createWebHistory } from 'vue-router'

import type { MessageNamespace } from '@/i18n'

import { pagePath, pagePathMatches } from '@shared/routing/page-routes'
import { pageRouteMetaFor } from '@shared/routing/route-meta'
import { decodedPathSegments, safeRedirectTarget } from '@shared/routing/safe-redirect'
import { normalizedRouteQueryValue } from '@shared/routing/route-query'
import { loginLocation, notFoundLocation, pageRouteNames } from './route-locations'

function lazyView(loader: () => Promise<{ default: Component }>) {
  return () => loader().then((module) => module.default)
}

function pageRoute(
  name: string,
  definition: Omit<RouteRecordSingleView, 'name' | 'path'>,
): RouteRecordRaw {
  return {
    ...definition,
    name,
    path: pagePath(name),
  }
}

const routes: RouteRecordRaw[] = [
  pageRoute(pageRouteNames.home, {
    component: lazyView(() => import('@/features/home/HomeView.vue')),
    meta: pageRouteMetaFor(pageRouteNames.home),
  }),
  pageRoute(pageRouteNames.login, {
    component: lazyView(() => import('@/features/auth/LoginView.vue')),
    meta: pageRouteMetaFor(pageRouteNames.login),
  }),
  pageRoute(pageRouteNames.import, {
    component: lazyView(() => import('@/features/import/ImportView.vue')),
    meta: pageRouteMetaFor(pageRouteNames.import),
  }),
  pageRoute(pageRouteNames.groups, {
    component: lazyView(() => import('@/features/groups/GroupsView.vue')),
    meta: pageRouteMetaFor(pageRouteNames.groups),
  }),
  pageRoute(pageRouteNames.groupDetail, {
    component: lazyView(() => import('@/features/groups/GroupDetailView.vue')),
    meta: pageRouteMetaFor(pageRouteNames.groupDetail),
  }),
  pageRoute(pageRouteNames.accessKeys, {
    component: lazyView(() => import('@/features/access-keys/AccessKeysView.vue')),
    meta: pageRouteMetaFor(pageRouteNames.accessKeys),
  }),
  pageRoute(pageRouteNames.monitor, {
    component: lazyView(() => import('@/features/monitor/MonitorView.vue')),
    meta: pageRouteMetaFor(pageRouteNames.monitor),
  }),
  pageRoute(pageRouteNames.schedule, {
    component: lazyView(() => import('@/features/monitor/ScheduleView.vue')),
    meta: pageRouteMetaFor(pageRouteNames.schedule),
  }),
  pageRoute(pageRouteNames.logs, {
    component: lazyView(() => import('@/features/logs/LogsView.vue')),
    meta: pageRouteMetaFor(pageRouteNames.logs),
  }),
  pageRoute(pageRouteNames.models, {
    component: lazyView(() => import('@/features/models/ModelsView.vue')),
    meta: pageRouteMetaFor(pageRouteNames.models),
  }),
  pageRoute(pageRouteNames.settings, {
    component: lazyView(() => import('@/features/settings/SettingsView.vue')),
    meta: pageRouteMetaFor(pageRouteNames.settings),
  }),
  {
    path: '/:pathMatch(.*)*',
    name: pageRouteNames.notFound,
    component: lazyView(() => import('@/features/not-found/NotFoundView.vue')),
    meta: {
      titleKey: 'notFound.title',
      requiresAuth: true,
    },
  },
]

export interface RouterAuth {
  hasCredential(): boolean
  getPrincipalType(): 'admin' | 'access_key' | null
}

export interface RouterMessages {
  loadNamespaces(namespaces: readonly MessageNamespace[]): Promise<void>
}

export function createAppRouter(
  auth: RouterAuth,
  history: RouterHistory = createWebHistory(),
  messages?: RouterMessages,
) {
  const router = createRouter({
    history,
    routes,
    sensitive: true,
    strict: true,
    scrollBehavior(to, from, savedPosition) {
      if (savedPosition) return savedPosition

      const pageViewChanged =
        to.path !== from.path ||
        normalizedRouteQueryValue(to.query.tab) !== normalizedRouteQueryValue(from.query.tab) ||
        normalizedRouteQueryValue(to.query.mode) !== normalizedRouteQueryValue(from.query.mode)
      return pageViewChanged ? { left: 0, top: 0 } : false
    },
  })
  router.beforeEach((to) => {
    if (
      typeof to.name === 'string' &&
      to.name !== pageRouteNames.notFound &&
      !pagePathMatches(to.name, to.path)
    ) {
      return notFoundLocation(decodedPathSegments(to.path))
    }
    if (to.meta.adminOnly && auth.getPrincipalType() === 'access_key') {
      return { name: pageRouteNames.home }
    }
    if (!to.meta.requiresAuth || auth.hasCredential()) {
      return true
    }
    return loginLocation(to.fullPath)
  })
  router.beforeResolve(async (to) => {
    const namespaces = (to.meta.messageNamespaces ?? []) as MessageNamespace[]
    await messages?.loadNamespaces(namespaces)
    return true
  })
  return router
}

export function safeRedirect(raw: unknown, router: Router): string {
  return safeRedirectTarget(raw, (value) => router.resolve(value), [
    pageRouteNames.login,
    pageRouteNames.notFound,
  ])
}
